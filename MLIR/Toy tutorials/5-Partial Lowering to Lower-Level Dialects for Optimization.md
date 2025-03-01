Reference:
[Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)

# **MLIR Toy Chapter 5**

## **Chapter 5: Partial Lowering to Lower-Level Dialects for Optimization**

The focus of this chapter is: **"We want to transform the Toy language into a lower-level MLIR dialect to leverage the optimization capabilities of the Affine dialect for mathematical operations."** However, we won't lower everything at once. Instead, we will perform **partial lowering**, allowing us to retain some high-level operations (e.g., `toy.print`) for later processing.

---

## **Why Perform Partial Lowering?**

We want to utilize the **optimization capabilities of the Affine dialect** to improve program performance. However, the Affine dialect does not support all operations, such as `toy.print`. Therefore, our approach is:

✅ Convert **computation-intensive parts** (e.g., mathematical operations) into **Affine, Arith, MemRef, and Func dialects**.

❌ Retain **`toy.print`** because it still needs to operate on TensorType. We will handle this in the next chapter.

---

## **MLIR's Dialect Conversion Mechanism**

MLIR provides a **Dialect Conversion Framework**, which helps transform illegal operations into legal ones. This framework requires three components (one of which is optional):

1️⃣ **Conversion Target**:
- Specify **which dialects are legal** (e.g., Affine, Arith, Func, MemRef).
- Specify **which dialects are illegal** (we mark Toy as illegal, but `toy.print` is an exception).

2️⃣ **Rewrite Patterns**:
- Define how to **convert illegal operations into legal ones** (e.g., transform `toy.transpose` into Affine loops).

3️⃣ **(Optional) Type Converter**:
- If variable types need to be converted (e.g., TensorType → MemRefType), this can be used. However, we don't need it this time.

---

## **Setting the Conversion Target**

Our goal is to **transform computations in the Toy language into the Affine dialect while temporarily retaining `toy.print`**. Here is the C++ code for setting the conversion target:

```cpp
void ToyToAffineLoweringPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  // Specify legal dialects
  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         func::FuncDialect, memref::MemRefDialect>();

  // Mark Toy dialect as illegal
  target.addIllegalDialect<ToyDialect>();

  // Except for `toy.print`, which remains legal as long as its operands are not TensorType
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return type.isa<TensorType>(); });
  });
}
```

### **What Does This Code Do?**

✔️ **Affine, Arith, Func, and MemRef** are legal dialects and can be used.

❌ **The Toy dialect is marked as illegal**, meaning all operations starting with `toy.` must be transformed.

✔️ **`toy.print` remains legal, but only if its input variables are not TensorType** (because we plan to convert TensorType → MemRefType).

---

## **Rewrite Patterns**

Next, we define how to **lower** illegal operations into legal ones. Here is an example of how **`toy.transpose`** is transformed into Affine loops:

```cpp
struct TransposeOpLowering : public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TransposeOp::getOperationName(), 1, ctx) {}

  llvm::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Convert to Affine loops
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::PatternRewriter &rewriter,
              ArrayRef<mlir::Value> memRefOperands,
              ArrayRef<mlir::Value> loopIvs) {
          TransposeOpAdaptor transposeAdaptor(memRefOperands);
          mlir::Value input = transposeAdaptor.input();

          // Use reversed indices for transposition
          SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return rewriter.create<mlir::AffineLoadOp>(loc, input, reverseIvs);
        });
    return success();
  }
};
```

### **What Does This Code Do?**

1️⃣ This is a **rewrite pattern** that tells MLIR **how to convert `toy.transpose` into Affine loops**.

2️⃣ It uses **Affine loops + Affine Load operations** to manually simulate the effect of `transpose`.

3️⃣ The benefit is that we can now leverage the optimization capabilities of the Affine dialect.

---

## **Performing Partial Lowering**

Once the conversion target and rewrite patterns are defined, we can **execute the conversion**:

```cpp
void ToyToAffineLoweringPass::runOnOperation() {
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();
}
```

This line of code means: **"Perform the conversion. If there are any unconverted illegal operations, fail."**

---

## **Result After Partial Lowering**

Let's look at the **difference in the Toy program before and after lowering**.

### **Before Lowering (Toy Language)**

```
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0) : tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

### **After Lowering (Converted to Affine)**

```
func.func @main() {
  %1 = memref.alloc() : memref<3x2xf64>
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %2 = affine.load %1[%arg1, %arg0] : memref<2x3xf64>
      %3 = arith.mulf %2, %2 : f64
      affine.store %3, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }
  toy.print %0 : memref<3x2xf64>
}
```

We can observe:
✅ `toy.transpose` is converted into Affine loops.

✅ `toy.mul` becomes `arith.mulf`.

✅ `toy.print` is retained (but now operates on `memref`).

---

## **Conclusion**

This chapter introduced **partial lowering**, aiming to **leverage the optimization capabilities of the Affine dialect** while **temporarily retaining `toy.print`**. In the next chapter, we will continue lowering `toy.print` to LLVM IR, enabling the entire program to generate actual machine code.

This approach of **partial lowering allows us to optimize mathematical operations while enabling different levels of dialects to coexist**, showcasing the power of MLIR! 🚀