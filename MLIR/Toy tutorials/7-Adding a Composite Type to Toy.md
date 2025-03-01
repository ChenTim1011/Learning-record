Reference:
[Chapter 7: Adding a Composite Type to Toy](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/)

# **MLIR Toy Chapter 7**

## **Chapter 7: Adding Composite Types to the Toy Language**

The core goal of this chapter is to add a new composite type to the Toy language—**StructType**. This allows Toy to define types containing multiple variables and supports the initialization, access, and manipulation of structures. These features require modifications to the frontend syntax, MLIR representation, operations, and optimization strategies.

---


## **1. Defining Structures in Toy**

The Toy language introduces the `struct` keyword to define structures. A structure can contain multiple variables, but these variables do not have initial values or shape information. The syntax is as follows:

```
struct MyStruct {
  var a;
  var b;
}
```

These structures can be used as variables or function parameters:

```
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
  return transpose(value.a) * transpose(value.b);
}

def main() {
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};
  var c = multiply_transpose(value);
  print(c);
}
```

Here:

- `Struct value = { ... }` initializes the structure using curly braces.
- `value.a` and `value.b` access structure members using the `.` operator.

---

## **2. Defining Structures in MLIR**

MLIR does not have a built-in type suitable for structures, so we need to define a custom **`StructType`** to represent structures. This type serves as a container for a set of element types and does not retain syntactic-level information such as variable names.

---

## **3. Defining the Type Class**

### **(1) Creating TypeStorage**

MLIR `Type` is immutable, so any type with additional parameters requires a corresponding **`TypeStorage`** to manage the uniqueness of these parameters.

```cpp
struct StructTypeStorage : public mlir::TypeStorage {
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  llvm::ArrayRef<mlir::Type> elementTypes;
};
```

Here:

- `KeyTy` defines the uniqueness condition, which is `llvm::ArrayRef<mlir::Type>`—a list of element types.
- `operator==` ensures that the same element types do not create multiple `StructTypeStorage` instances.
- `construct` is responsible for allocating `StructTypeStorage` using the `allocator`.

---

### **(2) Creating the StructType Class**

```cpp
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               StructTypeStorage> {
public:
  using Base::Base;

  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, elementTypes);
  }

  llvm::ArrayRef<mlir::Type> getElementTypes() { return getImpl()->elementTypes; }

  size_t getNumElementTypes() { return getElementTypes().size(); }
};
```

Here:

- The `get` method ensures that there is at least one element type and guarantees the uniqueness of `StructType` in the `MLIRContext`.
- `getElementTypes()` and `getNumElementTypes()` provide interfaces to access the internal types.

We need to register this type in `ToyDialect`:

```cpp
void ToyDialect::initialize() {
  addTypes<StructType>();
}
```

---

## **4. Exposing to ODS**

To make MLIR ODS (Operation Definition Specification) aware of our new type, we need to update the `tablegen` definition:

```
def Toy_StructType :
    DialectType<Toy_Dialect, CPred<"$_self.isa<StructType>()">,
                "Toy struct type">;

def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;
```

This allows `StructType` to be used in operation definitions.

---

## **5. Parsing and Printing StructType**

### **(1) Parsing**

Override `parseType` in `ToyDialect` to parse `StructType`:

```cpp
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  SmallVector<mlir::Type, 1> elementTypes;
  do {
    SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    if (!elementType.isa<mlir::TensorType, StructType>()) {
      parser.emitError(typeLoc, "element type must be a TensorType or StructType");
      return Type();
    }
    elementTypes.push_back(elementType);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}
```

### **(2) Printing**

```cpp
void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  StructType structType = type.cast<StructType>();
  printer << "struct<";
  llvm::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
```

This ensures that `StructType` is displayed as `struct<...>` in MLIR.

---

## **6. Operating on StructType**

### **(1) toy.struct_constant**

```
%0 = toy.struct_constant [
  dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
] : !toy.struct<tensor<*xf64>>
```

This represents the creation of a constant structure.

### **(2) toy.struct_access**

```
%1 = toy.struct_access %0[0] : !toy.struct<tensor<*xf64>> -> tensor<*xf64>
```

This represents accessing the `0`-th element from a `StructType`.

---

## **7. Optimizing StructType**

Now we can optimize `StructType`, such as performing **constant folding**.

When functions are inlined, MLIR can further simplify operations, removing unnecessary accesses.

---

This chapter introduced:

1. **Syntax**: Defining and accessing structures in Toy.
2. **Type System**: Representing `StructType` in MLIR.
3. **Parsing and Printing**: Parsing `StructType` into MLIR and converting it back to text format.
4. **Operations**: Adding `struct_constant` and `struct_access` operations.
5. **Optimization**: Simplifying `StructType`-related operations using constant folding.

These changes enhance the expressiveness of the Toy language, enabling it to support more complex data structures and operations.