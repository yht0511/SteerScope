# Upstream

The answer normalization in `math_equivalence.py` is vendored from
`hendrycks/math` commit `985bdc1696e88e8643f081a0ff4719da39f2ae2a`, file
`modeling/math_equivalence.py` (MIT). `util.py` preserves the official
last-balanced-box extraction semantics and adds a small outer-box remover.
