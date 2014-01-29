/*
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors:
 *      Adam Rak <adam.rak@streamnovation.com>
 */

#include "pipe/p_defines.h"
#include "pipe/p_state.h"
#include "pipe/p_context.h"
#include "util/u_blitter.h"
#include "util/u_double_list.h"
#include "util/u_transfer.h"
#include "util/u_surface.h"
#include "util/u_pack_color.h"
#include "util/u_memory.h"
#include "util/u_inlines.h"
#include "util/u_framebuffer.h"
#include "r600_shader.h"
#include "r600_pipe.h"
#include "r600_formats.h"
#include "compute_memory_pool.h"
#include "evergreen_compute.h"
#include "evergreen_compute_internal.h"
#include <inttypes.h>

/**
 * Creates a new pool
 */
struct compute_memory_pool* compute_memory_pool_new(
	struct r600_screen * rscreen)
{
	struct compute_memory_pool* pool = (struct compute_memory_pool*)
				CALLOC(sizeof(struct compute_memory_pool), 1);
	if (pool == NULL)
		return NULL;

	COMPUTE_DBG(rscreen, "* compute_memory_pool_new()\n");

	pool->screen = rscreen;
	return pool;
}

static void compute_memory_pool_init(struct compute_memory_pool * pool,
	unsigned initial_size_in_dw)
{

	COMPUTE_DBG(pool->screen, "* compute_memory_pool_init() initial_size_in_dw = %ld\n",
		initial_size_in_dw);

	pool->shadow = (uint32_t*)CALLOC(initial_size_in_dw, 4);
	if (pool->shadow == NULL)
		return;

	pool->next_id = 1;
	pool->size_in_dw = initial_size_in_dw;
	pool->bo = (struct r600_resource*)r600_compute_buffer_alloc_vram(pool->screen,
							pool->size_in_dw * 4);
	pool->fragmented = 0;
}

/**
 * Frees all stuff in the pool and the pool struct itself too
 */
void compute_memory_pool_delete(struct compute_memory_pool* pool)
{
	COMPUTE_DBG(pool->screen, "* compute_memory_pool_delete()\n");
	free(pool->shadow);
	if (pool->bo) {
		pool->screen->b.b.resource_destroy((struct pipe_screen *)
			pool->screen, (struct pipe_resource *)pool->bo);
	}
	free(pool);
}

/**
 * Searches for an empty space in the pool, return with the pointer to the
 * allocatable space in the pool, returns -1 on failure.
 */
int64_t compute_memory_prealloc_chunk(
	struct compute_memory_pool* pool,
	int64_t size_in_dw)
{
	struct compute_memory_item *item;

	int last_end = 0;

	assert(size_in_dw <= pool->size_in_dw);

	COMPUTE_DBG(pool->screen, "* compute_memory_prealloc_chunk() size_in_dw = %ld\n",
		size_in_dw);

	for (item = pool->item_list; item; item = item->next) {
		if (item->start_in_dw > -1) {
			if (item->start_in_dw-last_end > size_in_dw) {
				return last_end;
			}

			last_end = item->start_in_dw + item->size_in_dw;
			last_end += (1024 - last_end % 1024);
		}
	}

	if (pool->size_in_dw - last_end < size_in_dw) {
		return -1;
	}

	return last_end;
}

/**
 *  Search for the chunk where we can link our new chunk after it.
 */
struct compute_memory_item* compute_memory_postalloc_chunk(
	struct compute_memory_pool* pool,
	int64_t start_in_dw)
{
	struct compute_memory_item* item;

	COMPUTE_DBG(pool->screen, "* compute_memory_postalloc_chunck() start_in_dw = %ld\n",
		start_in_dw);

	/* Check if we can insert it in the front of the list */
	if (pool->item_list && pool->item_list->start_in_dw > start_in_dw) {
		return NULL;
	}

	for (item = pool->item_list; item; item = item->next) {
		if (item->next) {
			if (item->start_in_dw < start_in_dw
				&& item->next->start_in_dw > start_in_dw) {
				return item;
			}
		}
		else {
			/* end of chain */
			assert(item->start_in_dw < start_in_dw);
			return item;
		}
	}

	assert(0 && "unreachable");
	return NULL;
}

/**
 * Reallocates pool, conserves data.
 * @returns -1 if it fails, 0 otherwise
 */
int compute_memory_grow_pool(struct compute_memory_pool* pool,
	struct pipe_context * pipe, int new_size_in_dw)
{
	COMPUTE_DBG(pool->screen, "* compute_memory_grow_pool() "
		"new_size_in_dw = %d (%d bytes)\n",
		new_size_in_dw, new_size_in_dw * 4);

	assert(new_size_in_dw >= pool->size_in_dw);

	if (!pool->bo) {
		compute_memory_pool_init(pool, MAX2(new_size_in_dw, 1024 * 16));
		if (pool->shadow == NULL)
			return -1;
	} else {
		new_size_in_dw += 1024 - (new_size_in_dw % 1024);

		COMPUTE_DBG(pool->screen, "  Aligned size = %d (%d bytes)\n",
			new_size_in_dw, new_size_in_dw * 4);

		compute_memory_shadow(pool, pipe, 1);
		pool->shadow = realloc(pool->shadow, new_size_in_dw*4);
		if (pool->shadow == NULL)
			return -1;

		pool->size_in_dw = new_size_in_dw;
		pool->screen->b.b.resource_destroy(
			(struct pipe_screen *)pool->screen,
			(struct pipe_resource *)pool->bo);
		pool->bo = (struct r600_resource*)r600_compute_buffer_alloc_vram(
							pool->screen,
							pool->size_in_dw * 4);
		compute_memory_shadow(pool, pipe, 0);
	}

	return 0;
}

/**
 * Copy pool from device to host, or host to device.
 */
void compute_memory_shadow(struct compute_memory_pool* pool,
	struct pipe_context * pipe, int device_to_host)
{
	struct compute_memory_item chunk;

	COMPUTE_DBG(pool->screen, "* compute_memory_shadow() device_to_host = %d\n",
		device_to_host);

	chunk.id = 0;
	chunk.start_in_dw = 0;
	chunk.size_in_dw = pool->size_in_dw;
	chunk.prev = chunk.next = NULL;
	compute_memory_transfer(pool, pipe, device_to_host, &chunk,
				pool->shadow, 0, pool->size_in_dw*4);
}

/**
 * Allocates pending allocations in the pool
 * @returns -1 if it fails, 0 otherwise
 */
int compute_memory_finalize_pending(struct compute_memory_pool* pool,
	struct pipe_context * pipe)
{
	struct compute_memory_item *item, *next;
	struct compute_memory_item *last_item;

	int64_t allocated = 0;
	int64_t unallocated = 0;

	int64_t start_in_dw = 0;

	int err = 0;

	COMPUTE_DBG(pool->screen, "* compute_memory_finalize_pending()\n");

	for (item = pool->item_list; item; item = item->next) {
		COMPUTE_DBG(pool->screen, "  + list: offset = %i id = %i size = %i "
			"(%i bytes)\n",item->start_in_dw, item->id,
			item->size_in_dw, item->size_in_dw * 4);
	}

	/* allocated is the sum of all the item' size rounded
	 * up to a multiple of 1024 */
	for (item = pool->item_list; item; item = item->next) {
		allocated += item->size_in_dw;
		allocated += 1024 - (allocated % 1024);
	}

	/* unallocated is the sum of all the unallocated item'
	 * size rounded up to a multiple of 1024 */
	for (item = pool->unallocated_list; item; item = item->next) {
		unallocated += item->size_in_dw;
		unallocated += 1024 - (unallocated % 1024);
	}

	if (pool->fragmented)
		compute_memory_defrag(pool,pipe);

	/* allocated + unallocated is the size that all the items
	 * will use in the buffer, so we can grow the pool just
	 * one time */
	if (pool->size_in_dw < allocated+unallocated) {
		err = compute_memory_grow_pool(pool, pipe, allocated+unallocated);
		if (err == -1)
			return -1;
	}

	last_item = pool->item_list_end;
	/* This is equivalent to
	 * start_in_dw = allocated;
	 * but it's much clear this way */
	if (last_item) {
		start_in_dw = last_item->start_in_dw + last_item->size_in_dw;
		start_in_dw+= 1024 - (start_in_dw % 1024);
	}
	else {
		start_in_dw = 0;
	}

	/* Loop through the list of unallocated items, adding them
	 * to the end of the list of allocated items */
	for (item = pool->unallocated_list; item; item = next) {
		next = item->next;
		assert(start_in_dw + item->size_in_dw < pool->size_in_dw);

		item->start_in_dw = start_in_dw;
		item->next = NULL;
		item->prev = NULL;

		if (last_item) {
			last_item->next = item;
			item->prev = last_item;
		}
		else {
			pool->item_list = item;
		}

		last_item = item;
		start_in_dw += item->size_in_dw;
		start_in_dw += 1024 - (start_in_dw % 1024);
	}

	pool->item_list_end = last_item;
	pool->unallocated_list = NULL;
	pool->unallocated_list_end = NULL;

	return 0;
}

void compute_memory_defrag(struct compute_memory_pool *pool,
	struct pipe_context *pipe)
{
	struct compute_memory_item *item;
	int64_t last_pos;

	struct pipe_resource *gart = (struct pipe_resource *)pool->bo;
	struct pipe_transfer *xfer;
	uint32_t *map;

	map = pipe->transfer_map(pipe, gart, 0, PIPE_TRANSFER_READ,
			&(struct pipe_box) { .width = pool->size_in_dw * 4,
			.height = 1, .depth = 1 }, &xfer);
	assert(xfer);
	assert(map);
	memcpy(pool->shadow, map, pool->size_in_dw*4);
	pipe->transfer_unmap(pipe, xfer);

	last_pos = 0;
	for (item = pool->item_list; item; item = item->next) {
		if (item->start_in_dw != last_pos) {
			assert(item->start_in_dw > last_pos);
			memmove(pool->shadow + last_pos, pool->shadow + item->start_in_dw,
						item->size_in_dw * 4);
			item->start_in_dw = last_pos;
		}
		last_pos = last_pos + item->size_in_dw;
		last_pos += 1024 - (last_pos % 1024);
	}

	map = pipe->transfer_map(pipe, gart, 0, PIPE_TRANSFER_WRITE,
			&(struct pipe_box) { .width = pool->size_in_dw * 4,
			.height = 1, .depth = 1 }, &xfer);
	assert(xfer);
	assert(map);
	memcpy(map , pool->shadow, pool->size_in_dw*4);
	pipe->transfer_unmap(pipe, xfer);

	pool->fragmented = 0;
}

void compute_memory_free(struct compute_memory_pool* pool, int64_t id)
{
	struct compute_memory_item *item, *next;

	COMPUTE_DBG(pool->screen, "* compute_memory_free() id + %ld \n", id);

	for (item = pool->item_list; item; item = next) {
		next = item->next;

		if (item->id == id) {
			if (item->prev) {
				item->prev->next = item->next;
			}
			else {
				pool->item_list = item->next;
			}

			if (item->next) {
				item->next->prev = item->prev;
				pool->fragmented = 1;
			}
			else {
				pool->item_list_end = item->prev;
			}

			free(item);

			return;
		}
	}

	/* If unallocated items can't be freed, then this code
	 * isn't necesary at all */
	for (item = pool->unallocated_list; item; item = next) {
		next = item->next;

		if (item->id == id) {
			if (item->prev) {
				item->prev->next = item->next;
			}
			else {
				pool->unallocated_list = item->next;
			}

			if (item->next) {
				item->next->prev = item->prev;
			}
			else {
				pool->unallocated_list_end = item->prev;
			}

			free(item);

			return;
		}
	}

	fprintf(stderr, "Internal error, invalid id %"PRIi64" "
		"for compute_memory_free\n", id);

	assert(0 && "error");
}

/**
 * Creates pending allocations
 */
struct compute_memory_item* compute_memory_alloc(
	struct compute_memory_pool* pool,
	int64_t size_in_dw)
{
	struct compute_memory_item *new_item = NULL, *last_item = NULL;

	COMPUTE_DBG(pool->screen, "* compute_memory_alloc() size_in_dw = %ld (%ld bytes)\n",
			size_in_dw, 4 * size_in_dw);

	new_item = (struct compute_memory_item *)
				CALLOC(sizeof(struct compute_memory_item), 1);
	if (new_item == NULL)
		return NULL;

	new_item->size_in_dw = size_in_dw;
	new_item->id = pool->next_id++;
	new_item->pool = pool;

	last_item = pool->unallocated_list_end;

	if (last_item) {
		last_item->next = new_item;
		new_item->prev = last_item;
	}
	else {
		pool->unallocated_list = new_item;
	}
	pool->unallocated_list_end = new_item;

	COMPUTE_DBG(pool->screen, "  + Adding item %p id = %u size = %u (%u bytes)\n",
			new_item, new_item->id, new_item->size_in_dw,
			new_item->size_in_dw * 4);
	return new_item;
}

/**
 * Transfer data host<->device, offset and size is in bytes
 */
void compute_memory_transfer(
	struct compute_memory_pool* pool,
	struct pipe_context * pipe,
	int device_to_host,
	struct compute_memory_item* chunk,
	void* data,
	int offset_in_chunk,
	int size)
{
	int64_t aligned_size = pool->size_in_dw;
	struct pipe_resource* gart = (struct pipe_resource*)pool->bo;
	int64_t internal_offset = chunk->start_in_dw*4 + offset_in_chunk;

	struct pipe_transfer *xfer;
	uint32_t *map;

	assert(gart);

	COMPUTE_DBG(pool->screen, "* compute_memory_transfer() device_to_host = %d, "
		"offset_in_chunk = %d, size = %d\n", device_to_host,
		offset_in_chunk, size);

	if (device_to_host) {
		map = pipe->transfer_map(pipe, gart, 0, PIPE_TRANSFER_READ,
			&(struct pipe_box) { .width = aligned_size * 4,
			.height = 1, .depth = 1 }, &xfer);
		assert(xfer);
		assert(map);
		memcpy(data, map + internal_offset, size);
		pipe->transfer_unmap(pipe, xfer);
	} else {
		map = pipe->transfer_map(pipe, gart, 0, PIPE_TRANSFER_WRITE,
			&(struct pipe_box) { .width = aligned_size * 4,
			.height = 1, .depth = 1 }, &xfer);
		assert(xfer);
		assert(map);
		memcpy(map + internal_offset, data, size);
		pipe->transfer_unmap(pipe, xfer);
	}
}

/**
 * Transfer data between chunk<->data, it is for VRAM<->GART transfers
 */
void compute_memory_transfer_direct(
	struct compute_memory_pool* pool,
	int chunk_to_data,
	struct compute_memory_item* chunk,
	struct r600_resource* data,
	int offset_in_chunk,
	int offset_in_data,
	int size)
{
	///TODO: DMA
}
