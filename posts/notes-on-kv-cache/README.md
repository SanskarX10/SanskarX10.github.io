# kv_cache:

process prompt / context > sample additional tokens one by one > perform attention > need of key value pairs for each item in sequence > store these pairs in kv cache to avoid re computation for past tokens

t1, 						## k,v cached ##
t1, t2,						## k,v cached ##
t1, t2, t3					## k,v cached ##
t1, t2, t3, t4				## k,v cached ##
t1, t2, t3, t4, t5
t1, t2, t3, t4, t5, t6..

* per token, number of bytes stored => 2 * 2 * num_layers * n_heads * dim_head - [1]
	- 2 vectors k, value
	- 16 bits assumption so ( 2 bytes = 16 bits)

* weights will be multiplied by token embeddings 
	- W_k , W_v -> (d_model , d_model)
	- token_e -> (1, d_model)

* so FLOPS to get k,v for all of our layers will be : 2 * 2 * num_layers * (d_model ^ 2) - [2]
	- 2 * num_layers as two operations for k and v then repeat that for n layers
	- rest d_model^2 comes from [1]

~ flops in matmul : 2mn for (m,n) * (n, 1) , 2mnp for (m,n) * (n, p) : 2 for two operations of * and + in a matmul

so for a 52b params model, where d_model is 8192 and num_layers is 64, flops are
	- 2 * 2 * 64 * 8192 * 8192 = 17,179,869,184

for A100 gpu, where 312e12 FLOPs / sec is speed and 1.5e12 bytes/sec is memory bandwidth, for anthropic 52b:
	- for_memory : 17,179,869,184 / 1,500,000,000,000 =  0.0114532 seconds
	- for_compute : 17,179,869,184 / 312,000,000,000,000. = 0.000055 seconds
	- dividing these (memory/compute) would give us the number 208

~ FLOPS vs memory boundedness
	- need to load weight in memory -> costs memory bandwidth
	- flop bound : nothing is passed into the memory
	- memory bound : no floperations are happening 

for above hardware the ratio is 208, that mean time taken for k, v computation is equal for 1 or 208 tokens , token < 208 = memory bound, token > 208 = FLOPS bound

		^				   - 	
		|             compute
		|          -
time	|- - - - - - - - memory - -          intersection of this diagram is 208
		|     - 
		|   - 
		| - 
		------------------------------>
			num context tokens ( batch size) 
			
for a full forward pass, and use rest of the weights, a factor of 6 gets added in both numerator and denominator side (????)

so for full 52b model, the calculation will be " 6 * for_compute ~= 0.69 seconds" (from [2]) for upto 208 tokens (divide by n if using mutiple gpus) (kv cache computation time ∝ context length)
	- this 6 comes from 3 q, k, v matmul operations + 2 linear layer operations + 1 output projection and 2 for each multiplication and accumulation (not exact)

kv cache calculation is only 1/6th of work, forward pass cheap due to parellalization , sampling is opposite 
for one sample step, we have [2] flops
for full pass, we have 6 * [2] flops 
thus (not constant) 1/6th of flops * num_tokens can be saved at each steps ; since kv cache stores info for all sequences, as we sample generate seq_len keeps increasing, so work needed is a fraction of original work needed
without kv_cache it will be quadratic

capacity:

kv_cache amd weights are stored in GPU 
lets say A100 has 40gb VRAM

given the model params , to get size in bytes, mutiply params by 2 , 52 billion * 2 = 104e12 =~ 104 gb

so if we had 3 gpus, 120gb of vram , we have 16 gb left for kv_cache, according to [1] the bytes we needs for storage per token are, 2 * 2 * 64 * 8192 ( n_heads * dim_head = d_model) = 2,097,152 bytes ~= 0.002GB per token
16 / 0.002 -= 8000 tokens can fit into kv_cache for a single gpu ( divide by batch size to get num_tokens for each batch )










	
	



