Reference:
[Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)

### **Toy Tutorial**  

This tutorial demonstrates how to implement a basic toy language using MLIR. It introduces MLIR concepts, focusing on dialects, transformations, and lowering to LLVM or other backends. Inspired by the LLVM **Kaleidoscope** tutorial, it requires MLIR to be built beforehand.  

#### **Chapters Overview:**  
1. **Toy Language & AST** – Define the abstract syntax tree (AST).  
2. **AST to MLIR Dialect** – Emit MLIR using base MLIR concepts and attach semantics.  
3. **High-Level Optimizations** – Use pattern rewriting for optimizations.  
4. **Generic Transformations** – Apply dialect-independent transformations with interfaces.  
5. **Partial Lowering** – Convert high-level semantics to affine dialects for optimization.  
6. **Lowering to LLVM** – Generate LLVM IR and explore the lowering framework.  
7. **Extending Toy** – Add custom composite types and integrate them into the pipeline.  

