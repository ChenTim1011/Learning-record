Reference:
[Chapter 6: Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)

# **MLIR Toy Chapter 6**

### **Chapter 6: Lowering to LLVM and Code Generation**

In the previous chapter, we used the **Dialect Conversion Framework** to convert most of the **Toy Dialect** operations into **Affine Loop Nests**, enabling optimizations. In this chapter, we will further lower these operations into the **LLVM Dialect**, ultimately generating LLVM IR and using a **JIT (Just-In-Time Compiler)** to execute the generated code.

---

## **What is Lowering?**

Lowering is the process of transforming high-level **MLIR IR** into lower-level **LLVM IR**, allowing it to be processed by the **LLVM Backend** and ultimately generating machine code or performing JIT compilation. We will use the **Dialect Conversion Framework** to achieve this.

In MLIR, lowering is typically not done in a single step but through **multiple stages**:

1. **From Toy Dialect to Affine Dialect** (done in the previous chapter).
2. **From Affine, Arith, and Std Dialects to LLVM Dialect** (the focus of this chapter).
3. **From LLVM Dialect to LLVM IR**, which can then be used for JIT or compiled into machine code.

---

## **Step 1: Handling `toy.print`**

Currently, we have converted most Toy language operations into the Affine Dialect, but **`toy.print`** remains unhandled. We will transform it into a non-affine loop and call **`printf`** on each element to output the values.

During the MLIR conversion process, we don't need to directly generate LLVM Dialect instructions. Instead, we can use **transitive lowering**, first converting `toy.print` into structured loops and then further lowering it to the LLVM Dialect through the dialect conversion mechanism.

### **Defining `printf`**

In LLVM IR, we need to declare the **`printf`** function because MLIR does not have `printf`. Therefore, we need to manually insert it.

```cpp
/// If `printf` is not yet defined, insert its declaration
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module,
                                           LLVM::LLVMDialect *llvmDialect) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get("printf", context);

  // Define the type of the printf function
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                /*isVarArg=*/true);

  // Insert the printf function declaration
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
  return SymbolRefAttr::get("printf", context);
}
```

This code ensures that **`printf`** exists in the LLVM Dialect, allowing `toy.print` to use it during lowering.

---

## **Step 2: Setting Up Lowering Components**

We now need to specify the **three key components** for the lowering process:

1. **Conversion Target**: Specify that we want to convert to the **LLVM Dialect**.
2. **Type Converter**: Convert **MemRef** types into **LLVM-compatible types**.
3. **Conversion Patterns**: Provide specific lowering rules.

---

### **1. Conversion Target**

We want to convert **everything** (except `ModuleOp`) into the LLVM Dialect.

```cpp
mlir::ConversionTarget target(getContext());
target.addLegalDialect<mlir::LLVMDialect>();
target.addLegalOp<mlir::ModuleOp>();
```

This code means:

- **`addLegalDialect<mlir::LLVMDialect>()`**: Allow the use of the LLVM Dialect.
- **`addLegalOp<mlir::ModuleOp>()`**: Allow `ModuleOp` to exist (as it is the top-level scope).

---

### **2. Type Converter**

We use **`LLVMTypeConverter`** to handle the conversion of MLIR types into LLVM IR.

```cpp
LLVMTypeConverter typeConverter(&getContext());
```

This converter helps us handle:

- **`MemRef` → `LLVM Struct`**
- **`Index` → `i64`**
- **`Function Types` → `LLVM Function Types`**

---

### **3. Conversion Patterns**

We need to provide a set of conversion patterns to lower different dialects:

```cpp
mlir::RewritePatternSet patterns(&getContext());
mlir::populateAffineToStdConversionPatterns(patterns, &getContext());
mlir::cf::populateSCFToControlFlowConversionPatterns(patterns, &getContext());
mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
mlir::cf::populateControlFlowToLLVMConversionPatterns(patterns, &getContext());

// Toy Dialect-specific `PrintOp` conversion
patterns.add<PrintOpLowering>(&getContext());
```

Here, we do the following:

- **Affine → Std**: Convert **Affine Dialect** to standard operations (Std).
- **SCF → CFG**: Convert **SCF (Structured Control Flow)** to **Control Flow**.
- **Arith → LLVM**: Convert **arithmetic operations** to **LLVM Dialect**.
- **Func → LLVM**: Convert **FunctionOps** to **LLVM Dialect**.
- **ControlFlow → LLVM**: Handle the conversion of the `cf` dialect.

Finally, we add the **Toy Dialect's `PrintOp` lowering**.

---

## **Step 3: Performing Full Lowering**

Once we have set up the conversion target, type converter, and conversion patterns, we can perform the lowering:

```cpp
mlir::ModuleOp module = getOperation();
if (mlir::failed(mlir::applyFullConversion(module, target, patterns)))
  signalPassFailure();
```

This code attempts to convert the entire **Module** into the LLVM Dialect, ensuring that **no illegal operations** remain.

---

## **Step 4: Outputting LLVM IR**

After completing the MLIR conversion, we now need to convert the **LLVM Dialect** into **LLVM IR** and execute it using JIT.

```cpp
std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module);
if (!llvmModule)
  /* ... handle error ... */
```

This code converts the **LLVM Dialect** into **LLVM IR**.

---

## **Step 5: Setting Up JIT**

We can use the **MLIR Execution Engine** to execute the generated LLVM IR.

```cpp
auto maybeEngine = mlir::ExecutionEngine::create(module);
assert(maybeEngine && "JIT creation failed");

auto &engine = maybeEngine.get();
auto invocationResult = engine->invoke("main");
if (invocationResult) {
  llvm::errs() << "JIT execution failed\n";
  return -1;
}
```

This allows us to **directly execute the generated LLVM IR** and observe the results!

---

## **Conclusion**

In this chapter, we learned:

1. **How to use the Dialect Conversion Framework to lower MLIR to the LLVM Dialect**.
2. **How to define `printf` to make `toy.print` work**.
3. **How to convert different dialects and ultimately generate LLVM IR**.
4. **How to use JIT to execute the generated code**.

This is a critical part of implementing an MLIR-based compiler. In the next chapter, we will handle **structured data (Structs)**!