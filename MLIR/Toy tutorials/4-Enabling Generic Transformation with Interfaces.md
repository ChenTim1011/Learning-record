Reference:
[Chapter 4: Enabling Generic Transformation with Interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)

# **MLIR Toy Chapter 4**

## **Chapter 4: Enabling Generic Transformation with Interfaces**

This section explains how to use **interfaces** in MLIR to achieve generalized expression transformations, focusing on **inlining** and **shape inference**. These techniques allow us to avoid writing specific pattern-matching and rewriting rules for each operation. Instead, we can reuse transformation logic through generic interfaces, making the optimization process more extensible and reusable.

---

## 1. Background and Problem Motivation

In previous chapters, we introduced two methods for transforming specific operations:

1. **Directly writing pattern-matching and rewriting logic in C++** (e.g., for `transpose` or `reshape`).
2. **Using table-driven declarative rewrite rules (DRR)** via the **Operation Definition Specification (ODS)** to auto-generate pattern-matching and rewriting code.

While these methods work for individual operations, they cannot directly reuse logic for similar transformation requirements. To address this, MLIR provides the **interface mechanism**, which allows dialects and operations to provide generic information to optimization and analysis tools, enabling **generic expression transformations**.

For example, we can create an interface to handle inlining and shape inference, allowing all qualifying operations to be transformed using the same generic algorithm without writing individual rules for each operation.

---

## 2. Example: `multiply_transpose` in the Toy Language

Consider the following Toy language example, which defines a function `multiply_transpose`:

```
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

In this example, `multiply_transpose` first transposes two tensors and then performs element-wise multiplication. Since the Toy language's IR uses generic tensor types (e.g., `tensor<*xf64>`, indicating unknown shapes), optimizations and subsequent code generation face challenges (e.g., not knowing the actual shapes of tensors).

To address this, we take two steps:

1. **Inlining Pass**: Inline function calls into the call site, eliminating function definitions and leaving only the IR within the main function.
2. **Shape Inference Pass**: Infer concrete shapes based on known types, converting generic tensors into tensors with fixed shapes, facilitating further optimizations and code generation.

Both steps utilize MLIR's interface mechanism. Below, we explain the implementation details and code for each.

---

## 3. Inlining — Using Dialect and Operation Interfaces

### 3.1 Purpose and Challenges of Inlining

In our example, `multiply_transpose` is a small function. Without inlining, the overhead of function calls (e.g., preparing for calls and returns) might outweigh the function's computational cost. Therefore, we want to inline such small functions into the main function to improve runtime efficiency.

Additionally, since function definitions use generic tensors (e.g., `tensor<*xf64>`), but call sites know the concrete shapes (e.g., `tensor<2x3xf64>`), inlining must handle type mismatches.

### 3.2 Implementing `DialectInlinerInterface`

MLIR provides the **DialectInlinerInterface**, which we can inherit in our custom Toy dialect and override necessary methods. Below is a simplified example of defining the inlining interface for the Toy dialect:

```cpp
/// Define the inlining interface for the Toy dialect, inheriting from DialectInlinerInterface.
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Check if a given callable operation can be inlined into a call.
  // In Toy, all calls are legal to inline, so return true.
  bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final {
    return true;
  }

  // Check if a given operation can be inlined into a region.
  // In Toy, all operations can be inlined, so return true.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // Check if a source region can be inlined into a destination region.
  // In Toy, any function can be inlined, so return true.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned, IRMapping &valueMapping) const final {
    return true;
  }

  // Handle terminator operations (e.g., toy.return) after inlining by replacing return values.
  void handleTerminator(Operation *op, MutableArrayRef<Value> valuesToRepl) const final {
    // Only handle toy.return operations.
    auto returnOp = cast<ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// When inlining encounters type mismatches (e.g., tensor types at call sites vs. function definitions),
  /// this hook generates an explicit cast operation (CastOp).
  Operation *materializeCallConversion(OpBuilder &builder, Value input, Type resultType, Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};
```

In this interface, besides basic inlining legality checks, we override the `materializeCallConversion` method. This method inserts a **CastOp** when there is a type mismatch between the call site and the inlined function, ensuring successful inlining.

### 3.3 Operation Interfaces: `CallOpInterface` and `CallableOpInterface`

Inlining also requires identifying which operations represent function calls and which represent function definitions. MLIR provides finer-grained **Operation interfaces**:

- **CallOpInterface**: Marks call-like operations, such as `toy.generic_call`.
- **CallableOpInterface**: Marks callable-like operations, such as `toy.func`.

These interfaces are typically introduced in the TableGen Operation definition file (e.g., Ops.td) using the `DeclareOpInterfaceMethods` directive:

```
// Include CallOpInterface definitions (from mlir/Interfaces/CallInterfaces.td).
include "mlir/Interfaces/CallInterfaces.td"

// Define the function operation and add CallableOpInterface.
def FuncOp : Toy_Op<"func",
    [DeclareOpInterfaceMethods<CallableOpInterface>]> {
  ...
}

// Define the generic call operation and add CallOpInterface.
def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...
}
```

In the C++ implementation, we provide the necessary method definitions for these interfaces, such as:

- Implementing `getCallableRegion()` in `FuncOp` to return a pointer to the function body (Region).
- Implementing `getCallableForCallee()`, `setCalleeFromCallable()`, and `getArgOperands()` in `GenericCallOp` to allow the inliner to identify the call target and arguments.

This enables MLIR's inlining pass to correctly inline `multiply_transpose` into the `main` function.

### 3.4 Defining `CastOp`

Since inlining may encounter type mismatches (e.g., `tensor<2x3xf64>` at call sites vs. `tensor<*xf64>` in function definitions), a **CastOp** is needed to explicitly convert between types. Below is an example TableGen definition for `CastOp`:

```
def CastOp : Toy_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    Pure,
    SameOperandsAndResultShape
  ]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked,
    then shape is required to match. The operation is invalid if converting
    to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```

In C++, we implement the `areCastCompatible` method to verify compatibility between input and output types:

```cpp
bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  TensorType input = inputs.front().dyn_cast<TensorType>();
  TensorType output = outputs.front().dyn_cast<TensorType>();
  if (!input || !output || input.getElementType() != output.getElementType())
    return false;
  // If both are ranked, their shapes must match exactly.
  return !input.hasRank() || !output.hasRank() || input == output;
}
```

During inlining, when type mismatches occur, the `ToyInlinerInterface` calls `materializeCallConversion`, inserting a `CastOp` for explicit type conversion.

### 3.5 Result After Inlining

After inlining and necessary cast operations, the IR from all functions is embedded into the `main` function, as shown below (simplified):

```
toy.func @main() {
  %0 = toy.constant dense<[[1,2,3], [4,5,6]]> : tensor<2x3xf64>
  %1 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %2 = toy.transpose(%1 : tensor<*xf64>) to tensor<*xf64>
  ...  // Other operations generated after inlining
  toy.print %... : tensor<*xf64>
  toy.return
}
```

Through inlining, we eliminate the overhead of function calls and resolve type mismatches using `CastOp`, enabling shape inference to operate within a single function.

---

## 4. Shape Inference — Using Operation Interfaces

### 4.1 Problem Background

After inlining, the IR contains both static (fixed-shape) and generic (dynamic-shape) operations. Since subsequent optimizations and code generation rely on concrete shape information, we need to infer and propagate tensor shapes, converting generic types into fixed-shape types.

### 4.2 Defining the `ShapeInference` Operation Interface

Since shape inference is closely tied to an operation's core logic, we want operations requiring shape inference to implement a common interface. Using the ODS framework, we define an operation interface named **ShapeInference**:

```
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    // Define an interface method `inferShapes` to infer and set output shapes.
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}
```

We then add this interface to operations requiring shape inference (e.g., `MulOp`, `TransposeOp`). For example, for the multiplication operation, we add it in the TableGen definition:

```
def MulOp : Toy_Op<"mul",
    [..., DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  ...
}
```

### 4.3 Implementing `inferShapes` for Operations

Each operation implementing the `ShapeInference` interface must provide an `inferShapes()` method. For `MulOp`, the output shape can simply be copied from one of the input operands (e.g., the left-hand side):

```cpp
/// Infer the output shape of the MulOp.
/// This method sets the result type to match the left-hand operand's type.
void MulOp::inferShapes() {
  getResult().setType(getLhs().getType());
}
```

### 4.4 Implementing the Shape Inference Pass

Shape inference is implemented as a pass operating on a single function (`FuncOp`). The main workflow is as follows:

1. **Build a Worklist**:
    - Collect all operations returning dynamic types (generic tensors) that require shape inference.
2. **Process the Worklist Iteratively**:
    - Iteratively process operations in the worklist. If all inputs of an operation are known (non-generic), call its `inferShapes()` method to infer the result type. The process completes when the worklist is empty.

In the pass, we use the following code to check if an operation supports the `ShapeInference` interface and call its `inferShapes()` method:

```cpp
// Assume `op` is the operation being processed.
LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");

// Dynamically check if the operation implements the ShapeInference interface.
if (ShapeInference shapeOp = dyn_cast<ShapeInference>(op)) {
  shapeOp.inferShapes();
} else {
  op->emitError("unable to infer shape of operation without shape inference interface");
  return signalPassFailure();
}
```

Finally, we define a `ShapeInferencePass` inheriting from `OperationPass<FuncOp>` and implement the above workflow in `runOnOperation()`:

```cpp
class ShapeInferencePass : public mlir::PassWrapper<ShapeInferencePass, OperationPass<FuncOp>> {
  void runOnOperation() override {
    FuncOp function = getOperation();
    // Build the worklist and perform shape inference...
    // (Details of worklist construction and traversal are omitted here.)
  }
};

// Helper function to create an instance of ShapeInferencePass.
std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
```

Add this pass to the PassManager:

```cpp
pm.addPass(mlir::createShapeInferencePass());
```

After running, the IR's generic types are converted into fixed-shape tensor types based on inference results. For example, the final IR might look like this:

```
toy.func @main() {
  %0 = toy.constant dense<[[1,2,3], [4,5,6]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
```

---

## 5. Summary

Using MLIR's interface mechanism, we can achieve generalized expression transformations and avoid writing repetitive pattern-matching and rewriting code for each operation. Specifically:

1. **Inlining**:
    - Use **DialectInlinerInterface** to define inlining rules for the Toy dialect.
    - Combine **Operation interfaces** (`CallOpInterface` and `CallableOpInterface`) to identify call and function operations.
    - Insert **CastOp** for explicit type conversion when type mismatches occur during inlining.
2. **Shape Inference**:
    - Define an **Operation interface** (`ShapeInference`) and require operations to implement the `inferShapes()` method.
    - Write a pass to traverse operations in a function and infer result shapes based on known input types.

This generic transformation approach allows us to handle high-level language expression transformations more flexibly. The architecture can be easily extended to other dialects, showcasing MLIR's strengths in reusability and extensibility.

The next chapters will cover lower-level code generation, mapping Toy language operations to low-level dialects for compute-intensive tasks.