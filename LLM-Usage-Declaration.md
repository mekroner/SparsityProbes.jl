# LLM Usage

### Session 3 July
URL: https://share.gemini.google/KV4yXs9bvUns

Question:
```
https://github.com/E-175/SpectralClustering.jl/tree/main/src
```
Question:
```
Can you suggest any performance improvements in julia files in this source code?

```

### Session 29 Jun
URL: https://share.gemini.google/oMWIz055v4EA

Question: 
```
 I am writing a julia project and want to build the docs, but I get this error (SparsityProbes) pkg> activate docs

  Activating project at `~/projects/JuML/SparsityProbes.jl/docs` [...]
```
Queston:
```
Error: 9 docstrings not included in the manual:

│ 

│     SparsityProbes._
[...]
```

### Session 21 Jun
URL: https://share.gemini.google/QKJp8QUlPgrP
Question: 
```
 I have the issue that my docs script is failing
name: Documentation 
```

### Session 23 Jun
URL: https://share.gemini.google/vWcTIyOlGWx0
Question: 
```
How can I use julia 11 using juliaup 
```
```
I have a manifest of the wrong julia version. Can I just delete it and use ] test to regenerate it? 
```

### Session 21 Jun

URL: https://share.gemini.google/KTI2bQkIJDpI

Question:
```
Take a look at this Julia code

 function ADTypes.jacobian_sparsity(f, x, detector::Union{ChunkedDetector, BloomFilterDetector})
        if detector isa ChunkedDetector
            return _jacobian_sparsity_chunked(f, x, detector.chunk_size)
        elseif detector isa BloomFilterDetector
            return _jacobian_sparsity_bloom_filter(f, x, detector.filter_size_m, detector.num_hashes_k)
        end
    end

I feel it would be better to have two functions one function ADTypes.jacobian_sparsity(f, x, ChunkedDetector) and ADTypes.jacobian_sparsity(f, x, BloomFilterDetector)
```

### Session 21 Jun
URL: https://share.gemini.google/El1BWkg7pkIu
Question:
```
how should jula docstrings look
```
