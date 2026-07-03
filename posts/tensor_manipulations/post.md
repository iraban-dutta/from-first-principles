---
title: Tensor Manipulations 101
slug: tensor_manipulations
published: 2026-07-03
revised: 2026-07-03
description: A library of notes on frequently encountered tensor manipulations
---

# Tensor Manipulations 101

**Published:** 3 July 2026

**Last Updated:** 3 July 2026

Tensor manipulations form the foundation of modern deep learning. Every operation in a neural network—from embedding lookups and attention mechanisms to convolutions and loss computation—ultimately boils down to selecting, reshaping, combining or broadcasting tensors.

This article is a collection of the tensor manipulation patterns that I repeatedly encounter while working with NumPy and PyTorch. The emphasis is on building intuition through examples rather than memorizing APIs.

---

{{TOC}}

---

# 1 Tensor Indexing

Tensor indexing is one of the most fundamental operations in scientific computing. Once you understand indexing, operations such as embedding lookups, batched inference and attention mechanisms become much easier to reason about.

Throughout this section, we'll use the following tensor.

```text
       col

       0   1   2   3

row 0  10  11  12  13
row 1  20  21  22  23
row 2  30  31  32  33
row 3  40  41  42  43
```

Its shape is

```python
(4, 4)
```

---

## 1.1 The Mental Model

For a 2D tensor, indexing always follows the pattern

```python
tensor[row_selector, column_selector]
```

Think of indexing as a two-step process.

1. Select the rows.
2. From those rows, select the columns.

Everything else—integer indexing, slicing, advanced indexing—is simply a variation of this idea.

For example,

```python
arr[1:3, 0:2]
```

can be mentally evaluated as

```text
Step 1

Keep rows 1 and 2

↓

20 21 22 23
30 31 32 33

↓

Step 2

Keep columns 0 and 1

↓

20 21
30 31
```

> **Mental Model**
>
> Always think **rows first, columns second**.

---

## 1.2 Integer Indexing

Selecting an integer removes that dimension.

Consider

```python
arr[2]
```

We select exactly one row.

```text
30 31 32 33
```

Shape

```python
(4,)
```

Notice that the row dimension has disappeared.

Instead of a matrix, we now have a single vector.

Similarly,

```python
arr[2, 1]
```

returns

```text
31
```

Shape

```python
()
```

Both dimensions disappear because we selected exactly one row and one column.

### Why does this happen?

A tensor dimension represents a collection of values.

When we specify an integer index, that collection collapses into one specific element along that axis.

There is no longer a dimension to represent.

> **Key Takeaways**
>
> - Integer indexing removes a dimension.
> - Selecting one row from `(4,4)` produces `(4,)`.
> - Selecting one row and one column produces a scalar.

---

## 1.3 Slice Indexing

Unlike integer indexing, slices preserve dimensions.

Suppose we write

```python
arr[2:3]
```

Result

```text
30 31 32 33
```

Although only one row is selected, the output shape is

```python
(1, 4)
```

not

```python
(4,)
```

The reason is that a slice always represents a range—even if that range contains only one element.

Similarly,

```python
arr[1:3]
```

returns

```text
20 21 22 23
30 31 32 33
```

Shape

```python
(2,4)
```

Notice how the row dimension is preserved.

> **Key Takeaways**
>
> - Slices preserve dimensions.
> - `2` and `2:3` are fundamentally different.
> - Use slices whenever you want to preserve tensor rank.

---

## 1.4 The Colon (`:`) Operator

The colon simply means

> **Select everything along this axis.**

Examples

```python
arr[:, 2]
```

All rows

Column 2

Result

```text
12
22
32
42
```

---

```python
arr[2, :]
```

Row 2

All columns

```text
30 31 32 33
```

---

```python
arr[:, :]
```

Entire tensor.

The colon does not change the tensor.

It merely states

> "Don't filter anything along this dimension."

> **Key Takeaways**
>
> - `:` means "everything".
> - It is commonly used to leave one axis unchanged while selecting another.
> - Most indexing expressions use `:` somewhere.

---


## 1.5 Continuous Submatrix Extraction

A very common operation is extracting a rectangular block from a tensor.

Suppose we want the following submatrix.

```text
21 22
31 32
```

Visually,

```text
       0   1   2   3

0      10  11  12  13
1      20 [21 22] 23
2      30 [31 32] 33
3      40  41  42  43
```

Notice that

- Rows are continuous (`1, 2`)
- Columns are continuous (`1, 2`)

Since both rows and columns form contiguous ranges, slicing is the natural choice.

```python
arr[1:3, 1:3]
```

Output

```text
21 22
31 32
```

Shape

```python
(2,2)
```

This follows directly from our mental model.

```text
Step 1

Keep rows 1 and 2

↓

20 21 22 23
30 31 32 33

↓

Step 2

Keep columns 1 and 2

↓

21 22
31 32
```

> **Rule of Thumb**
>
> If the desired rows and columns are contiguous, use slicing.

### Key Takeaways

- Continuous rows → use a slice.
- Continuous columns → use a slice.
- Continuous rectangular block → slice both dimensions.

---

## 1.6 Non-Continuous Submatrix Extraction

Now suppose we want

```text
20 22
40 42
```

Rows

```text
1, 3
```

Columns

```text
0, 2
```

These rows are **not contiguous**.

These columns are **not contiguous**.

Therefore, slices alone cannot represent this selection.

Many beginners expect the following to work.

```python
arr[[1,3], [0,2]]
```

Expected

```text
20 22
40 42
```

Unfortunately, this is **not** how NumPy or PyTorch interpret the expression.

---

## 1.7 Pairwise Indexing — The Most Common Gotcha

This is one of the most common interview questions and a frequent source of bugs.

When both dimensions receive index lists,

```python
arr[[1,3], [0,2]]
```

PyTorch and NumPy **pair the indices together**.

Think of it as

```python
zip([1,3], [0,2])
```

The coordinate pairs become

```text
(1,0)

(3,2)
```

which correspond to

```text
20

42
```

Result

```python
tensor([20,42])
```

Shape

```python
(2,)
```

Notice what did **not** happen.

The framework never generated all possible row-column combinations.

Instead, it matched the indices pairwise.

Visualizing the lookup,

```text
       0   1   2   3

0      10  11  12  13

1     [20] 21  22  23

2      30  31  32  33

3      40  41 [42] 43
```

This behaviour is identical in both NumPy and PyTorch.

> **Mental Model**
>
> Multiple index lists behave like coordinate pairs—not row and column sets.

---

### Extracting a True Non-Continuous Submatrix

To obtain

```text
20 22
40 42
```

perform the indexing in two steps.

First,

select the rows.

```python
arr[[1,3], :]
```

Result

```text
20 21 22 23
40 41 42 43
```

Shape

```python
(2,4)
```

Now,

select the desired columns.

```python
arr[[1,3], :][:, [0,2]]
```

Result

```text
20 22
40 42
```

Shape

```python
(2,2)
```

Thinking visually,

```text
Original

20 21 22 23
40 41 42 43

↓

Keep columns 0 and 2

↓

20 22
40 42
```

Although this looks like two indexing operations,

this is the correct way to express

> "Give me all combinations of these rows and these columns."

### NumPy Alternative

NumPy additionally provides

```python
np.ix_()
```

which constructs the Cartesian product of row and column indices.

```python
arr[np.ix_([1,3], [0,2])]
```

Output

```text
20 22
40 42
```

PyTorch does **not** have a direct equivalent of `np.ix_`, so the two-step approach remains the clearest and most portable solution.

### Key Takeaways

- Multiple index lists do **not** generate a submatrix.
- They are interpreted as coordinate pairs.
- Think `zip(rows, cols)`, **not** all row-column combinations.
- For a non-contiguous submatrix:
    1. Select rows.
    2. Then select columns.

---

## 1.8 Shape Rules

By now, a clear pattern should have emerged.

The type of index determines how the output shape changes.

| Index Type | Shape Behaviour | Example |
|------------|-----------------|---------|
| Integer | Removes a dimension | `arr[2] → (4,)` |
| Slice | Preserves a dimension | `arr[2:3] → (1,4)` |
| List / Tensor | Size of dimension becomes number of indices | `arr[[1,3]] → (2,4)` |

These three rules explain the vast majority of indexing operations encountered in practice.

### Section Summary

✔ Integer indexing removes dimensions.

✔ Slice indexing preserves dimensions.

✔ `:` means "everything" along that axis.

✔ Continuous submatrices are naturally expressed using slices.

✔ Multiple index lists represent **pairwise coordinates**, not submatrices.

✔ Non-contiguous submatrices should be extracted by selecting rows first and columns second.


# 2 Advanced Indexing

So far, our index has either been

- an integer,
- a slice, or
- a one-dimensional list.

What happens if the index itself is a tensor?

This seemingly simple idea powers one of the most fundamental operations in deep learning—**embedding lookup**.

Throughout this section, consider the tensor

```python
A.shape == (8,8)
```

Each row contains eight values.

---

## 2.1 Single Row Lookup

Suppose we index with a single integer.

```python
A[3]
```

The output is

```python
shape = (8,)
```

We obtain the fourth row.

Nothing surprising here—this is simply integer indexing.

```text
A

Row 0

Row 1

Row 2

Row 3  ← Selected

↓

Output

(8,)
```

### Key Takeaways

- Scalar index selects one row.
- Integer indexing removes one dimension.
- Output shape is simply the remaining dimensions.

---

## 2.2 Multiple Row Lookup

Now suppose we index with several rows.

```python
A[[0,2,5]]
```

Instead of one row,

PyTorch returns all requested rows stacked together.

Output

```python
shape = (3,8)
```

Visualize it as

```text
Input indices

0

2

5

↓

Row 0

Row 2

Row 5

↓

Stack

↓

(3,8)
```

Notice that

```python
3
```

comes directly from the number of indices supplied.

The original tensor has not changed.

We have simply gathered three rows from it.

Exactly the same behaviour exists in NumPy.

### Key Takeaways

- List indexing gathers multiple rows.
- Number of selected rows equals the number of indices.
- Shape becomes

```python
(number_of_indices, remaining_dimensions)
```

---

## 2.3 Batched Row Lookup

Now comes the important idea.

Instead of a one-dimensional index tensor,

suppose the indices themselves have shape

```python
(2,3)
```

For example,

```python
indices =

[[0,2,3],

 [2,5,7]]
```

Performing

```python
A[indices]
```

produces

```python
shape = (2,3,8)
```

Why?

Think of each row of the index tensor independently.

Batch 0

```text
0

2

3

↓

Three rows

↓

(3,8)
```

Batch 1

```text
2

5

7

↓

Three rows

↓

(3,8)
```

Finally,

both batches are stacked together.

```text
Batch 0

(3,8)

Batch 1

(3,8)

↓

(2,3,8)
```

Notice something interesting.

The output shape begins with

```python
(2,3)
```

which is exactly the shape of the index tensor.

The remaining dimension

```python
8
```

comes from the original tensor.

### Key Takeaways

- Index tensors can themselves be multi-dimensional.
- Every index becomes one lookup.
- The shape of the index tensor is preserved in the output.

---

## 2.4 The Golden Rule

After seeing several examples, a very useful rule emerges.

> **Output Shape = Index Shape + Remaining Tensor Dimensions**

This single rule explains almost every advanced indexing operation.

Examples

| Tensor Shape | Index Shape | Output Shape |
|--------------|-------------|--------------|
| `(8,8)` | `()` | `(8,)` |
| `(8,8)` | `(3,)` | `(3,8)` |
| `(8,8)` | `(2,3)` | `(2,3,8)` |
| `(50000,768)` | `(4,)` | `(4,768)` |
| `(50000,768)` | `(32,128)` | `(32,128,768)` |

Whenever you're unsure about an indexing expression,

first determine

1. the shape of the index tensor,
2. the remaining dimensions of the source tensor.

Concatenate them together.

That is almost always the output shape.

### Key Takeaways

- Learn the golden rule instead of memorizing examples.
- Output shape = Index Shape + Remaining Tensor Dimensions.
- This rule directly explains embedding layers.

---

## 2.5 Why Embedding Layers Work

Suppose

```python
E.shape == (50000,768)
```

where

- 50000 = vocabulary size
- 768 = embedding dimension

Now consider a token batch.

```python
tokens.shape == (32,128)
```

where

- 32 = batch size
- 128 = sequence length

Running

```python
E[tokens]
```

produces

```python
(32,128,768)
```

No magic happened.

PyTorch simply performed advanced indexing.

Every integer inside

```python
tokens
```

was used to look up one row of

```python
E
```

The batch dimension

```python
32
```

was preserved.

The sequence dimension

```python
128
```

was preserved.

The embedding dimension

```python
768
```

came from the original embedding table.

Thinking visually,

```text
Embedding Table

(V,D)

↓

Index Tensor

(B,T)

↓

Output

(B,T,D)
```

Once viewed through the lens of tensor indexing,

an embedding layer is nothing more than a batched row lookup.

### Section Summary

✔ Integer indexing selects one row.

✔ List indexing selects multiple rows.

✔ Tensor indexing performs batched row lookups.

✔ Output Shape = Index Shape + Remaining Tensor Dimensions.

✔ `nn.Embedding` is fundamentally an advanced indexing operation.

# 3 Embedding Layers

An embedding layer converts a discrete token into a dense vector representation.

For example,

```python
Vocabulary Size = 27

Embedding Dimension = 2
```

The embedding layer stores one 2-dimensional vector for every vocabulary token.

Conceptually,

```text
Token ID

↓

Embedding Table

↓

Embedding Vector
```

The important question is:

> **How does the embedding layer retrieve the correct row?**

There are two equivalent ways to think about it.

---

## 3.1 One-Hot Encoding + Matrix Multiplication

Suppose our vocabulary size is

```python
V = 27
```

and the embedding dimension is

```python
D = 2
```

We begin by representing the token as a one-hot vector.

For example,

```text
Token = 3

↓

[0 0 0 1 0 0 ... 0]
```

Shape

```python
(27,)
```

The embedding matrix has shape

```python
W.shape == (27,2)
```

The embedding is computed as

```python
X_emb = X_inp.T @ W
```

Shape

```text
(1 × 27)

×

(27 × 2)

↓

(1 × 2)
```

But something much more interesting is happening.

Since the one-hot vector contains exactly one value equal to one,

every row except one gets multiplied by zero.

Suppose

```text
Token = 3
```

Then

```text
0 × Row 0

0 × Row 1

0 × Row 2

1 × Row 3

0 × Row 4

...
```

Everything disappears except

```text
Row 3
```

The matrix multiplication simply returns

```text
Row 3 of W
```

Nothing more.

Nothing less.

> **Mental Model**
>
> One-hot encoding + matrix multiplication is simply a row lookup.

---

### Example

Suppose

```text
Embedding Matrix

Token

0 → [0.2 0.8]

1 → [1.3 0.1]

2 → [2.5 1.7]

3 → [4.0 3.2]
```

If

```text
Token = 2
```

then

```python
X.T @ W
```

returns

```text
[2.5 1.7]
```

which is exactly the third row.

### Key Takeaways

- One-hot vectors contain exactly one non-zero entry.
- Matrix multiplication keeps only one row.
- OHE + Linear Layer = Row Lookup.

---

## 3.2 Integer Lookup (`nn.Embedding`)

Instead of constructing a one-hot vector,

PyTorch stores the same embedding matrix but directly indexes it.

Suppose

```python
E.shape == (27,2)
```

Now,

instead of

```python
X.T @ W
```

we simply write

```python
E[token]
```

If

```python
token = 2
```

then

```python
E[2]
```

returns

```text
[2.5 1.7]
```

which is exactly the same vector obtained using one-hot encoding.

Notice that no one-hot vector was ever created.

PyTorch directly performs a row lookup.

Thinking visually,

```text
Token ID

↓

Row Number

↓

Embedding Matrix

↓

Selected Row
```

Exactly the same operation.

### Why is this preferred?

Creating a one-hot vector of length

```python
50000
```

for every token is wasteful.

Almost every value is zero.

Instead,

PyTorch simply jumps directly to the desired row.

This saves

- memory,
- computation,
- and time.

---

## 3.3 Why Both Methods Are Equivalent

At first glance,

these two pieces of code appear completely different.

**Method 1**
```python
X_emb = X.T @ W
```

**Method 2**
```python
X_emb = E[token]
```

In reality,

they perform exactly the same operation.

| One-Hot Encoding | Embedding Lookup |
|------------------|------------------|
| Build one-hot vector | Use integer index |
| Multiply with embedding matrix | Directly index embedding matrix |
| Select row *i* | Select row *i* |
| Output embedding vector | Output embedding vector |

The only difference is **how the row is accessed**.

One method computes it indirectly through matrix multiplication.

The other retrieves it directly through indexing.

Mathematically,

both produce identical embeddings.

This is why `nn.Embedding` can be viewed as nothing more than an optimized implementation of one-hot encoding followed by a linear layer without bias.

### Connection to the Previous Chapter

Recall the Golden Rule.

```
Output Shape

=

Index Shape

+

Remaining Tensor Dimensions
```

Suppose

```python
tokens.shape == (32,128)
```

and

```python
E.shape == (50000,768)
```

Then

```python
E[tokens]
```

produces

```python
(32,128,768)
```

because

```text
Index Shape

(32,128)

+

Remaining Dimensions

(768)

↓

(32,128,768)
```

The embedding layer is simply advanced indexing applied to the embedding matrix.

### Section Summary

✔ Every token corresponds to one row of the embedding matrix.

✔ One-hot encoding selects that row through matrix multiplication.

✔ `nn.Embedding` selects the same row through indexing.

✔ Both methods produce identical embeddings.

✔ `nn.Embedding` is faster because it skips constructing sparse one-hot vectors.


# 4 Reshaping Tensors

Reshaping changes **how we interpret the data**, not the data itself.

This is an important distinction.

Whenever possible, PyTorch simply changes the tensor's metadata instead of moving values in memory.

Throughout this section, we'll use

```python
x = torch.arange(12)

tensor([0, 1, 2, ..., 11])
```

Shape

```python
(12,)
```

---

## 4.1 Reshaping

Suppose we reshape

```python
x.reshape(3,4)
```

Result

```text
0  1  2  3
4  5  6  7
8  9 10 11
```

Shape

```python
(3,4)
```

Notice that

- no values changed,
- no values were reordered.

Only our interpretation of the data changed.

Similarly,

```python
x.reshape(2,2,3)
```

produces

```python
shape = (2,2,3)
```

The elements still appear in exactly the same order.

### Mental Model

Think of reshaping as pouring the same water into a differently shaped container.

The water doesn't change.

Only the container does.

### Key Takeaways

- `reshape()` changes the shape.
- It never changes element order.
- Total number of elements must remain the same.

---

## 4.2 Flattening

Flattening converts a higher-dimensional tensor into a one-dimensional tensor.

Suppose

```python
A =
```

```text
1 2
3 4
```

Shape

```python
(2,2)
```

Flattening produces

```python
tensor([1,2,3,4])
```

Shape

```python
(4,)
```

PyTorch provides several ways to achieve this.

### Method 1 — `reshape()`

```python
A.reshape(-1)
```

---

### Method 2 — `view()`

```python
A.view(-1)
```

---

### Method 3 — `flatten()`

```python
torch.flatten(A)
```

For most practical purposes,

all three produce

```python
tensor([1,2,3,4])
```

### Which should I use?

If your goal is simply

> "Convert everything into one dimension"

then

```python
torch.flatten()
```

is the most expressive.

If you're already reshaping tensors elsewhere,

```python
reshape(-1)
```

is perfectly acceptable.

### Key Takeaways

- Flattening is simply reshaping into one dimension.
- The element order remains unchanged.
- `flatten()`, `reshape(-1)` and `view(-1)` often produce identical results.

---

## 4.3 `view()` vs `reshape()`

These two functions often confuse beginners.

At first glance,

they appear identical.

```python
x.view(3,4)
```

```python
x.reshape(3,4)
```

In many cases,

they produce exactly the same output.

The difference lies in **memory**.

### The Mental Model

Internally,

PyTorch stores tensor values as one long one-dimensional block of memory.

For example,

```python
A =
```

```text
1 2
3 4
```

is actually stored as

```text
1 2 3 4
```

The tensor's shape,

number of dimensions,

and strides simply tell PyTorch how to interpret this memory.

Changing the shape often requires changing only this metadata.

No data movement is necessary.

Visualize it as

```text
Underlying Storage

1 2 3 4 5 6 7 8

↓

View A

2 × 4

↓

View B

4 × 2

↓

View C

8
```

The underlying storage never changes.

Only the interpretation changes.

### `view()`

`view()` creates another interpretation of the same memory.

No new tensor is allocated.

This makes it extremely memory efficient.

However,

`view()` requires the tensor to occupy contiguous memory.

If the tensor is not contiguous,

`view()` raises an error.

---

### `reshape()`

`reshape()` is more flexible.

Whenever possible,

it behaves exactly like `view()`.

If the tensor is not contiguous,

PyTorch silently creates a new contiguous tensor before reshaping.

Because of this,

`reshape()` almost always succeeds.

### Practical Recommendation

Unless you specifically need the guarantees provided by `view()`,

prefer

```python
reshape()
```

It is more robust and easier to use.

PyTorch will automatically use a view whenever possible.

### NumPy Comparison

NumPy behaves similarly.

```python
np.reshape()
```

attempts to return a view whenever possible,

but may allocate new memory if necessary.

Thus,

the high-level intuition is the same in both libraries.

### Key Takeaways

- Tensor values are stored in one contiguous memory block.
- Shape is only metadata describing how that memory should be interpreted.
- `view()` never copies data.
- `view()` requires contiguous memory.
- `reshape()` uses a view when possible and copies only when necessary.

---

### Section Summary

✔ Reshaping changes interpretation, not values.

✔ Flattening is simply reshaping into one dimension.

✔ Tensor storage is fundamentally one-dimensional.

✔ `view()` is a new interpretation of existing memory.

✔ `reshape()` is the safer, more general API.


# 5 Combining Tensors

When working with tensors, we frequently need to combine multiple tensors into one.

Before choosing an operation, always ask yourself one question.

> **Am I making an existing dimension larger, or am I creating an entirely new dimension?**

Almost every tensor combination function can be understood through this lens.

Throughout this section, let

```python
A =

[[1,2],
 [3,4]]

B =

[[5,6],
 [7,8]]
```

Both tensors have shape

```python
(2,2)
```

---

## 5.1 `concatenate()` / `torch.cat()`

Concatenation joins tensors along an **existing dimension**.

No new dimension is introduced.

### Example 1 — Concatenate Rows

```python
np.concatenate([A,B], axis=0) # Numpy

torch.cat([A,B], dim=0) # PyTorch
```

Result

```text
1 2
3 4
5 6
7 8
```

Shape

```python
(4,2)
```

Notice that

only the row dimension became larger.

```text
Before

(2,2)

+

(2,2)

↓

After

(4,2)
```

---

### Example 2 — Concatenate Columns

```python
np.concatenate([A,B], axis=1) # Numpy

torch.cat([A,B], dim=1) # PyTorch
```

Result

```text
1 2 5 6
3 4 7 8
```

Shape

```python
(2,4)
```

This time,

the column dimension became larger.

### Mental Model

Concatenation **extends an existing axis**.

Nothing new is created.

### Key Takeaways

- `concatenate()` and `torch.cat()` are equivalent.
- Existing dimension grows.
- Tensor rank remains unchanged.

---

## 5.2 `stack()` / `torch.stack()`

Unlike concatenation,

stacking creates an **entirely new dimension**.

Instead of asking

> "Where should these values go?"

think

> "I need one more axis."

---

### Example 1 — Stack Along `axis=0`

```python
np.stack([A,B], axis=0) # Numpy

torch.stack([A,B], dim=0) # PyTorch
```

Result

```text
Layer 0

1 2
3 4

────────────

Layer 1

5 6
7 8
```

Shape

```python
(2,2,2)
```

Interpretation

```text
2 matrices

↓

Each matrix has

2 rows

↓

Each row has

2 columns
```

Notice

the new dimension became the outermost axis.

---

### Example 2 — Stack Along `axis=1`

```python
np.stack([A,B], axis=1) # Numpy

torch.stack([A,B], dim=1) # PyTorch
```

Shape

```python
(2,2,2)
```

Although the shape is identical,

the meaning is completely different.

Think row-wise.

```text
Row 0

[
 [1 2]

 [5 6]
]

Row 1

[
 [3 4]

 [7 8]
]
```

The new dimension was inserted between

rows

and

columns.

---

### Example 3 — Stack Along `axis=-1`

```python
np.stack([A,B], axis=-1) # Numpy

torch.stack([A,B], dim=-1) # PyTorch
```

Result

```text
[[1,5] [2,6]]

[[3,7] [4,8]]
```

Shape

```python
(2,2,2)
```

Now,

every element became a pair.

Instead of thinking

```text
One scalar
```

think

```text
One vector

↓

[1,5]
```

This is a very common operation when constructing feature vectors.

### Mental Model

`stack()` first creates a new dimension,

then places tensors along that dimension.

### Key Takeaways

- `stack()` always increases tensor rank by one.
- Same shape does **not** imply same meaning.
- The chosen axis determines where the new dimension is inserted.

---

## 5.3 `expand_dims()` / `unsqueeze()`

Sometimes we don't want to join tensors.

We simply want to add a dimension of size one.

NumPy

```python
np.expand_dims() # Numpy

torch.unsqueeze() # PyTorch
```

perform exactly this operation.

Suppose

```python
A.shape

(2,2)
```

Adding a leading dimension

```python
np.expand_dims(A, axis=0) # Numpy

A.unsqueeze(0) # PyTorch
```

produces

```python
(1,2,2)
```

Interpretation

```text
1 matrix

↓

2 rows

↓

2 columns
```

Adding a trailing dimension

```python
np.expand_dims(A, axis=-1) # Numpy

A.unsqueeze(-1) # PyTorch
```

produces

```python
(2,2,1)
```

This is particularly useful before broadcasting,

which we'll study in the next chapter.

### Key Takeaways

- `expand_dims()` and `unsqueeze()` are equivalent.
- They always insert a dimension of size one.
- They never modify the underlying data.

---

## 5.4 `squeeze()`

`squeeze()` performs the opposite operation.

It removes dimensions whose size equals one.

Suppose

```python
A.shape

(1,2,2,1)
```

Then

```python
A.squeeze()
```

produces

```python
(2,2)
```

Only singleton dimensions disappear.

Dimensions larger than one remain untouched.

### Mental Model

Think of

```python
unsqueeze()
```

and

```python
squeeze()
```

as inverse operations.

```text
(2,2)

↓

unsqueeze(0)

↓

(1,2,2)

↓

squeeze()

↓

(2,2)
```

### Key Takeaways

- `squeeze()` removes singleton dimensions.
- It never removes dimensions larger than one.
- It is the inverse of `unsqueeze()`.

---

### Section Summary

| Operation | Existing Dimension Grows? | New Dimension Created? |
|-----------|---------------------------|------------------------|
| `concatenate()` / `torch.cat()` | ✅ Yes | ❌ No |
| `stack()` / `torch.stack()` | ❌ No | ✅ Yes |
| `reshape()` | ❌ No | ❌ No |
| `expand_dims()` / `unsqueeze()` | ❌ No | ✅ Size = 1 |
| `squeeze()` | ❌ No | Removes Size-1 Dimension |

### Memory Rules

✔ `cat()` extends an existing dimension.

✔ `stack()` inserts a new dimension.

✔ `reshape()` changes only the interpretation.

✔ `unsqueeze()` inserts a singleton dimension.

✔ `squeeze()` removes singleton dimensions.


# 6 Broadcasting

Broadcasting allows NumPy and PyTorch to perform element-wise operations on tensors of different shapes **without explicitly copying data**.

Instead of physically expanding smaller tensors, the framework **pretends** that dimensions of size one have been stretched to match the larger tensor.

Broadcasting is one of the reasons tensor libraries are both expressive and efficient.

---

## 6.1 The Broadcasting Rules

Whenever two tensors participate in an element-wise operation (`+`, `-`, `*`, `/`, etc.), the following rules are applied.

### Rule 1 — Align Shapes from the Right

If the tensors have different ranks,

pad the smaller shape with leading dimensions of size one.

Example

```text
(2,3)

+

(3,)

↓

(2,3)

+

(1,3)
```

Notice that only the shape changes.

No data has been copied.

---

### Rule 2 — Compare Dimensions Right to Left

For every axis,

the dimensions are compatible if

- they are equal, or
- one of them is `1`.

Otherwise,

broadcasting fails.

---

### Rule 3 — Output Dimension = Maximum

For every axis,

the output dimension is simply

```text
max(dimA, dimB)
```

---

### Mental Algorithm

Whenever you're unsure,

follow the same four steps.

```text
Pad smaller shape with leading 1's

↓

Compare dimensions from RIGHT to LEFT

↓

Equal or One?

↓

Take the Maximum

↓

Otherwise → Error
```

This algorithm works for every broadcasting problem.

### Key Takeaways

- Broadcasting aligns shapes from the right.
- Missing dimensions are treated as size one.
- Dimensions must either match or equal one.
- Output dimension is the maximum along each axis.

---

## 6.2 Singleton Dimensions

A dimension whose size equals one is called a **singleton dimension**.

Singleton dimensions are special because they can be broadcast.

Suppose

```python
A.shape == (2,1)
```

```text
1

2
```

and

```python
B.shape == (1,3)
```

```text
10 20 30
```

Adding them

```python
A + B
```

produces

```text
11 21 31

12 22 32
```

Shape

```python
(2,3)
```

Thinking visually,

```text
Column Vector

1
2

↓

Broadcast Across Columns

↓

1 1 1
2 2 2
```

Similarly,

```text
Row Vector

10 20 30

↓

Broadcast Across Rows

↓

10 20 30
10 20 30
```

The result becomes

```text
11 21 31

12 22 32
```

Notice that

neither tensor was physically copied.

The framework simply behaves **as if** they had been expanded.

### Mental Model

Singleton dimensions are **stretchable**.

They behave like elastic bands that can expand to match larger dimensions whenever necessary.

### Key Takeaways

- Dimensions of size one can broadcast.
- Broadcasting behaves like stretching.
- No additional memory is allocated for the stretched tensor.

---

## 6.3 Canonical Examples

### Example 1

```text
(4,3)

+

(4,3)
```

Both tensors already have the same shape.

Output

```text
(4,3)
```

---

### Example 2

```text
(2,3)

+

(3,)
```

Pad

```text
(2,3)

+

(1,3)
```

Compare

```text
2 vs 1

↓

2

3 vs 3

↓

3
```

Output

```text
(2,3)
```

---

### Example 3

```text
(2,1)

+

(1,3)
```

Compare

```text
2 vs 1

↓

2

1 vs 3

↓

3
```

Output

```text
(2,3)
```

This is the row-vector plus column-vector example discussed earlier.

---

### Example 4

```text
(5,4,3)

+

(3,)
```

Pad

```text
(5,4,3)

+

(1,1,3)
```

Output

```text
(5,4,3)
```

The final dimension matches,

while the leading singleton dimensions expand automatically.

---

### Example 5

One of the classic interview questions.

```text
(8,1,6,1)

+

(7,1,5)
```

First,

pad the smaller tensor.

```text
(8,1,6,1)

+

(1,7,1,5)
```

Compare axis by axis.

```text
8 vs 1

↓

8

1 vs 7

↓

7

6 vs 1

↓

6

1 vs 5

↓

5
```

Final output shape

```text
(8,7,6,5)
```

Although this looks intimidating,

it follows exactly the same algorithm.

### Key Takeaways

- Every broadcasting problem follows the same four-step procedure.
- Large examples are simply repeated applications of the same rules.
- Never try to memorize special cases.

---

## 6.4 Invalid Broadcasting Examples

Broadcasting fails whenever two dimensions

- are different, and
- neither equals one.

### Example 1

```text
(4,3)

+

(4,)
```

Pad

```text
(4,3)

+

(1,4)
```

Compare

```text
3 vs 4
```

Neither equals one.

Broadcasting fails.

---

### Example 2

```text
(2,3)

+

(2,2)
```

Compare

```text
3 vs 2
```

Again,

neither dimension equals one.

Result

```text
ValueError
```

### Key Takeaways

Broadcasting fails whenever

```text
Dimension A ≠ Dimension B

AND

Dimension A ≠ 1

AND

Dimension B ≠ 1
```

---

## 6.5 Broadcasting in Deep Learning

Broadcasting appears constantly during neural network training.

For example,

suppose

```python
X.shape == (64,512)
```

and

```python
b.shape == (512,)
```

Computing

```python
X + b
```

produces

```python
(64,512)
```

The bias vector

```python
(512,)
```

is automatically broadcast across every sample in the batch.

Another common example is

```python
X.shape == (Batch, Sequence, Hidden)
```

combined with

```python
scale.shape == (Hidden,)
```

Again,

the scale vector is broadcast across every batch element and every token.

This is one of the reasons broadcasting is indispensable in deep learning.

### NumPy vs PyTorch

Broadcasting semantics are identical in NumPy and PyTorch.

Once you understand the rules,

they transfer directly between both libraries.

---

### Section Summary

### Broadcasting Checklist

✔ Align shapes from the right.

✔ Pad smaller shapes with leading ones.

✔ Dimensions are compatible if they are equal or one equals one.

✔ Output dimension is the maximum.

✔ Singleton dimensions are stretchable.

✔ Broadcasting never explicitly copies data.

✔ NumPy and PyTorch follow the same broadcasting rules.