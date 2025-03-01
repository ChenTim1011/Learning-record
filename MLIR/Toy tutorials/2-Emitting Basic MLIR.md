Reference:
[Chapter 2: Emitting Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)

# **MLIR Toy Chapter 2**

## **Chapter 2: Emitting Basic MLIR** 

This chapter introduces the core concepts and code examples for emitting basic MLIR (Multi-Level Intermediate Representation). It covers the design philosophy of MLIR's multi-level intermediate representation, how to interact with MLIR's interfaces, and how to define a dialect and its operations for the Toy language. Additionally, it explains how to use the **Operation Definition Specification (ODS)** framework to simplify the definition and implementation of operations.

---

## 1. Introduction to MLIR and Multi-Level Intermediate Representation

In traditional compilers (e.g., LLVM), there is typically a predefined set of types and (mostly low-level or RISC-like) instructions. The language frontend is responsible for performing language-specific type checking, analysis, and transformations before converting the results into LLVM IR. For example, in C++ with Clang, the AST is used to handle template instantiation, static analysis, and other transformations.

MLIR was designed to address these challenges. It provides a fully extensible infrastructure that allows frontends to define their own dialects and operations while retaining only a few core concepts (e.g., operations, attributes, types). This enables intermediate representations at different levels to interoperate without requiring the reimplementation of extensive infrastructure. As a result, multiple frontends can share the same foundational facilities, avoiding redundant work and enabling high-level analysis and transformations.

---

## 2. MLIR Interfaces and Opaque API

### 2.1 Concepts of Dialects and Operations

In MLIR, an **Operation** is the most basic unit of computation in the intermediate representation, similar to an instruction in LLVM. Each operation has fixed inputs (operands) and outputs (results), and it also includes:

- A unique name (e.g., `"toy.transpose"`), typically prefixed with a dialect namespace.
- A set of **Attributes**, which are immutable constant values such as booleans, integers, or strings.
- Type information describing the inputs and outputs (e.g., `tensor<2x3xf64>`).
- A **Location** for debugging and tracking the source of the operation.
- **Successors** (for control flow) and **Regions** (for structured operations like function definitions).

Additionally, MLIR uses **Static Single Assignment (SSA)**, where each operation's output value is unique and automatically assigned a prefixed name (e.g., `%t_tensor`). This name is valid during parsing but may not be preserved in the in-memory representation.

### 2.2 Opaque API

MLIR achieves high extensibility by allowing users to define custom IR elements (e.g., attributes, operations, types). However, all custom elements can be reduced to the basic concepts mentioned earlier. If a user does not register a specific dialect, MLIR can still parse, represent, and round-trip these operations. However, without semantic information, subsequent transformations and verifications can only rely on structural checks (e.g., dominance relationships). Therefore, in practical systems, it is recommended to register custom dialects and operations with the `MLIRContext` to enable verification, optimization, and more precise processing.

For example, the following IR can be parsed by MLIR even if the Toy dialect is not registered:

```
func.func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```

However, since the Toy dialect is not registered, MLIR cannot perform deeper semantic checks or optimizations on `toy.transpose`.

---

## 3. Defining the Toy Dialect

### 3.1 Defining a Dialect in C++

In C++, defining a dialect typically involves inheriting from the `mlir::Dialect` class and registering custom attributes, operations, and types in the constructor. For example, here is a C++ definition of the `ToyDialect`:

```cpp
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// A helper function to get the dialect namespace, which is "toy" here.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// Called in the constructor to register attributes, operations, and types.
  void initialize();
};
```

In the `initialize()` method, operations are registered using calls like `addOperations<...>()`.

### 3.2 Declarative Definition Using TableGen

MLIR also supports declarative definitions using TableGen, which simplifies boilerplate code and automatically generates documentation and APIs. For example, the Toy dialect can be defined in TableGen as follows:

```
// Define the 'toy' dialect with its namespace and description.
def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A high-level dialect for analyzing and optimizing the Toy language";
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];
  let cppNamespace = "toy";
}
```

The `mlir-tblgen` tool can then generate C++ declarations and definitions for the dialect, which can be loaded into the `MLIRContext` using `context.loadDialect<ToyDialect>();`.

---

## 4. Defining Toy Operations

### 4.1 Operations and Traits

In MLIR, each operation is represented by a C++ class. For example, the `ConstantOp` represents constant values in the Toy language. It has the following characteristics:

- **Zero Operands**: A constant operation does not depend on other values as inputs.
- **One Result**: It produces a single SSA value, typically of a tensor type.
- **Typed Result Accessor**: Provides a function to retrieve the result type.

Here is a simplified C++ definition of the `ConstantOp`:

```cpp
class ConstantOp : public mlir::Op<
                     ConstantOp,
                     mlir::OpTrait::ZeroOperands,
                     mlir::OpTrait::OneResult,
                     mlir::OpTrait::OneTypedResult<TensorType>::Impl> {
public:
  using Op::Op;

  // Define the operation name, prefixed with the dialect namespace.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  // Helper function to access the constant value from the attribute.
  mlir::DenseElementsAttr getValue();

  // Additional verification: Check that the result type matches the constant value.
  LogicalResult verifyInvariants();

  // Provide various build methods to create instances of this operation.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

The operation is registered in the `ToyDialect` initialization method:

```cpp
void ToyDialect::initialize() {
  addOperations<ConstantOp>();
}
```

### 4.2 Op and Operation

MLIR provides two levels of abstraction:

- **Operation**: A generic opaque class that provides APIs common to all operations but lacks operation-specific semantics.
- **Op Derived Classes**: For example, `ConstantOp` provides a type-safe wrapper for specific operations, offering dedicated accessors and methods. This design allows operations to be passed "by-value" and enables casting a generic `Operation` to a specific `Op` using LLVM's casting mechanism.

For example, given an `Operation*`, it can be cast as follows:

```cpp
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);
  if (!op)
    return;
  // Get the internal Operation pointer (note that 'op' is a smart pointer wrapper).
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation &&
         "these operation instances are the same");
}
```

### 4.3 Using the ODS Framework to Define Operations

In addition to manual C++ implementations, MLIR supports declarative operation definitions using the **Operation Definition Specification (ODS)**. With TableGen, the properties of an operation (e.g., inputs, outputs, descriptions, verification, and build methods) can be concisely specified, and corresponding C++ code is automatically generated.

For example, the `ConstantOp` can be defined in TableGen as follows:

```
def ConstantOp : Toy_Op<"constant"> {
  // Documentation: A brief summary and detailed description for auto-generated docs.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // Define the operation's inputs and outputs:
  // Input: An F64ElementsAttr (a 64-bit floating-point ElementsAttr).
  let arguments = (ins F64ElementsAttr:$value);
  // Output: An F64Tensor (a 64-bit floating-point tensor).
  let results = (outs F64Tensor);

  // Additional verification: Custom verify code (hasVerifier = 1 generates verify()).
  let hasVerifier = 1;

  // Define build methods to allow users to create the operation via builder.create<ConstantOp>(...).
  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build(builder, result, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];
}
```

Using the above TableGen definition, running `mlir-tblgen` generates the C++ declarations and definitions for `ConstantOp`. This approach is concise and ensures that operation definitions remain consistent with documentation, facilitating maintenance and auto-generated documentation.

---

## 5. Custom Assembly Format

In MLIR, each operation has a default generic assembly format, which displays all inputs, attributes, types, and locations. However, this format can be verbose and less readable for users. To address this, custom assembly formats can be defined for operations to make the generated IR more concise and intuitive.

For example, for `toy.print`, suppose we want the format to display only the input and its type, resulting in:

```
toy.print %5 : tensor<*xf64> loc(...)
```

In C++, we can override the operation's `print` and `parse` methods:

```cpp
// Define a custom print method.
void PrintOp::print(mlir::OpAsmPrinter &printer) {
  printer << "toy.print " << op.input();
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

// Define a custom parse method.
mlir::ParseResult PrintOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand inputOperand;
  mlir::Type inputType;
  if (parser.parseOperand(inputOperand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() ||
      parser.parseType(inputType))
    return mlir::failure();
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return mlir::failure();
  return mlir::success();
}
```

Alternatively, in TableGen, the `assemblyFormat` field can be used to declaratively specify the format:

```
def PrintOp : Toy_Op<"print"> {
  let arguments = (ins F64Tensor:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

This results in more concise and readable IR.

---

## 6. Complete Toy Example

Combining all the above parts, we can generate a complete Toy IR. For example, consider the following Toy program:

```
# User-defined generic function that operates on unknown-shaped arguments.
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

The generated IR might look like this (with location information):

```
module {
  "toy.func"() ({
  ^bb0(%arg0: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":4:1)):
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:10)
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":5:25)
    "toy.return"(%2) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":5:3)
  }) {sym_name = "multiply_transpose", type = (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":4:1)
  "toy.func"() ({
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00],
                                        [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>}
         : () -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00,
                                          4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>}
         : () -> tensor<6xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose}
         : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose}
         : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/Examples/Toy/Ch2/codegen.toy":12:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("test/Examples/Toy/Ch2/codegen.toy":13:3)
    "toy.return"() : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
  }) {sym_name = "main", type = () -> ()} : () -> () loc("test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
```

From this IR, we can observe:

- Each operation is prefixed with `toy.`, corresponding to the Toy dialect.
- Each operation includes complete input, attribute, type, and location information.
- Using ODS definitions, the generated operation classes provide convenient APIs, automatic verification, construction methods, and custom assembly formats.

Through this process, we successfully transformed the Toy language's AST (or high-level representation) into an MLIR-based intermediate representation. Subsequent optimizations, transformations, and code generation passes can then be applied to this IR.

---

## Summary

This chapter introduced how to emit basic MLIR IR by:

1. **Multi-Level Intermediate Representation**:
    - MLIR uses a highly simplified set of core concepts (operations, attributes, types, locations, etc.), enabling a unified and extensible intermediate representation for various high-level languages.
2. **Interacting with MLIR Interfaces**:
    - The **Dialect** mechanism groups custom language elements into independent namespaces.
    - Even unregistered operations can be parsed (Opaque API), but registration is recommended for verification and transformation.
3. **Defining the Toy Dialect and Operations**:
    - Defined dialects and operations using both C++ and TableGen (ODS framework), providing complete interfaces, verification, construction methods, and custom assembly formats.
    - The `ConstantOp` example demonstrated how to define inputs (attributes), outputs (results), verification logic, and construction methods.
4. **Relationship Between Op and Operation**:
    - MLIR separates operations into generic `Operation` and type-safe `Op` derived classes, facilitating subsequent transformations and operation access.
5. **Custom Assembly Format**:
    - Overriding C++ `print`/`parse` methods or defining `assemblyFormat` in TableGen makes the generated IR more readable.
6. **Complete Toy Example**:
    - From a Toy language example program to generating complete MLIR IR with function definitions, operation calls, and various attributes, types, and locations, the entire process was demonstrated.

