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

#ifndef COMPUTE_MEMORY_POOL
#define COMPUTE_MEMORY_POOL

#include <stdlib.h>

struct compute_memory_pool;

struct compute_memory_item
{
	int64_t id; ///ID of the memory chunk

	int untouched; ///True if the memory contains only junk, no need to save it for defrag

	int64_t start_in_dw; ///Start pointer in dwords relative in the pool bo
	int64_t size_in_dw; ///Size of the chunk in dwords

	struct compute_memory_pool* pool;

	struct compute_memory_item* prev;
	struct compute_memory_item* next;
};

struct compute_memory_pool
{
	int64_t next_id; ///For generating unique IDs for memory chunks
	int64_t size_in_dw; ///Size of the pool in dwords

	struct r600_resource *bo; ///The pool buffer object resource
	struct compute_memory_item* item_list; ///Allocated memory chunks in the buffer,they must be ordered by "start_in_dw"
	struct compute_memory_item* item_list_end; ///End of item_list
	struct compute_memory_item* unallocated_list; ///Unallocated memory chunks
	struct compute_memory_item* unallocated_list_end; ///End of unallocated_list
	struct r600_screen *screen;

	int fragmented; /// 1 if the pool buffer is fragmented

	uint32_t *shadow; ///host copy of the pool, used for defragmentation
};


struct compute_memory_pool* compute_memory_pool_new(struct r600_screen *rscreen); ///Creates a new pool
void compute_memory_pool_delete(struct compute_memory_pool* pool); ///Frees all stuff in the pool and the pool struct itself too

int compute_memory_grow_pool(struct compute_memory_pool* pool, struct pipe_context * pipe,
	int64_t allocated, int64_t unallocated);

void compute_memory_shadow(struct compute_memory_pool* pool,
	struct pipe_context * pipe, int device_to_host,
	int64_t allocated);

int compute_memory_finalize_pending(struct compute_memory_pool* pool,
	struct pipe_context * pipe);
void compute_memory_defrag(struct compute_memory_pool *pool,
	struct pipe_context *pipe, int64_t allocated);
void compute_memory_free(struct compute_memory_pool* pool, int64_t id);
struct compute_memory_item* compute_memory_alloc(struct compute_memory_pool* pool, int64_t size_in_dw); ///Creates pending allocations

void compute_memory_transfer(struct compute_memory_pool* pool,
	struct pipe_context * pipe, int device_to_host,
	struct compute_memory_item* chunk, void* data,
	int offset_in_chunk, int size);

void compute_memory_transfer_direct(struct compute_memory_pool* pool, int chunk_to_data, struct compute_memory_item* chunk, struct r600_resource* data, int offset_in_chunk, int offset_in_data, int size); ///Transfer data between chunk<->data, it is for VRAM<->GART transfers

#endif
