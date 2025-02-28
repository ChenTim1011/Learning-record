# ** Evaluation: MLIR Applications and Assessment**

MLIR's core goals are **generality** and **extensibility**, allowing various compiler projects to utilize it for building **high-performance intermediate representations (IRs)**. This section evaluates MLIR's impact through **community adoption** and **specific use cases**.

---

## **📌 1 TensorFlow Graphs (TF Computational Graphs)**

TensorFlow (TF) is a widely used machine learning framework whose internal computation graph is a **dataflow graph with dynamic execution semantics**, representing relationships between different computational nodes.

### **MLIR in TensorFlow:**

1. **As the Core of TensorFlow IR**
    - TensorFlow uses MLIR to model its internal IR.
    - Enables transformations and optimizations on IR.
    - Generates efficient execution code for different target devices (e.g., TPU, GPU, CPU).

2. **Supports Various IR Optimizations and Transformations**
    - **Algebraic Simplifications**
    - **Graph Transformations**
    - **Device Mapping** (mapping computational graphs to specific hardware, e.g., TPU)
    - **Lowering to lower-level IRs (such as XLA)**

3. **TensorFlow Graph Representation in MLIR:**
    
    ```
    %0 = tf.graph (%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : !tf.resource) {
      // Asynchronous execution; %control ensures execution order
      %1, %control = tf.ReadVariableOp(%arg2)
        : (!tf.resource) -> (tensor<f32>, !tf.control)
      %2, %control_1 = tf.Add(%arg0, %1)
        : (tensor<f32>, tensor<f32>) -> (tensor<f32>, !tf.control)
      %control_2 = tf.AssignVariableOp(%arg2, %arg0, %control)
        : (!tf.resource, tensor<f32>) -> !tf.control
      %3, %control_3 = tf.Add(%2, %arg1)
        : (tensor<f32>, tensor<f32>) -> (tensor<f32>, !tf.control)
      tf.fetch %3, %control_2 : tensor<f32>, !tf.control
    }
    ```
    
    **🔍 Breakdown of the MLIR Representation:**
    - **`tf.ReadVariableOp`** reads variable `%arg2`, producing `%1`.
    - **`tf.Add`** performs addition (`%arg0 + %1` → `%2`).
    - **`tf.AssignVariableOp`** assigns `%arg0` to variable `%arg2`.
    - **`tf.Add`** executes another addition (`%2 + %arg1` → `%3`).
    - **`tf.fetch`** outputs `%3` and control flow `%control_2`.

---

## **📌 Applications of TensorFlow Graphs in MLIR**

Using MLIR, TensorFlow computational graphs can be **transformed, optimized, and lowered** to different backends (such as XLA, TPU, GPU, and CPU), enhancing performance and adaptability across computing environments.

### **✅ Key Advantages**

1. **Graph Optimization**
    - Algebraic simplifications (e.g., eliminating identity operations).
    - Constant folding.
    - Avoiding redundant computations.

2. **Device-Specific Transformations**
    - Automatically assigns operations to **TPU, GPU, CPU**.
    - **Generates optimized computational graphs for different hardware**.

3. **Lowering to XLA for High-Performance Computation**
    - XLA (Accelerated Linear Algebra) is TensorFlow’s high-performance compiler.
    - MLIR facilitates the transformation of TensorFlow graphs into XLA IR.

---

# **📌 Summary**

✅ MLIR is widely adopted in academia and industry, with major projects like TensorFlow integrating it.

✅ TensorFlow leverages MLIR to **model computational graphs** and perform optimizations and device mapping.

✅ **Through MLIR, TensorFlow computational graphs can be lowered to various IRs, further optimizing execution performance!** 🚀

---

## **📌 2 Polyhedral Code Generation**

One of MLIR’s core objectives is **polyhedral code generation for accelerators**, which is a key motivation behind the design of the **Affine dialect**. The Affine dialect provides a **structured polyhedral representation** and supports **progressive lowering**, allowing the compiler to gradually transform IRs for different levels of optimization and hardware requirements.

### **📍 1 Similarities**

The Affine dialect uses **structured multi-dimensional types** to model memory access patterns, ensuring:

1. **Alias-Free Access in Memory Indexing**
    - A prerequisite for **polyhedral dependence analysis**.
    - Example: Ensuring `A[i]` and `A[j]` do not interfere.

2. **Affine Modeling Approach**
    - **Attributes** → Define affine maps and integer sets, allowing **compile-time determinability**.
    - **Ops** → Apply affine constraints, such as `affine.for` (Affine loops) and `affine.if` (Affine conditionals).

3. **Properties of Affine Loops**
    - **affine.for** → Loop bounds are determined by affine maps and must depend on **loop-invariant values**.
    - **affine.if** → Conditionals use affine integer sets for precise analysis.
    - **affine.load / affine.store** → Restrict indexing to **affine expressions**, ensuring accurate **loop dependence analysis**.

> 🚀 These properties make the Affine dialect more mathematically analyzable and optimization-friendly than traditional low-level IRs!

---

### **📍  2 Differences from Traditional Polyhedral Frameworks**

#### **(1) Rich Types**

- **MLIR provides structured memory reference types with layout maps:**

    ```
    %1 = affine.load %B[%arg1] : memref<? x f32, (d0)[s0] -> (d0 + s0)>
    ```

    - `memref` represents **memory references**.
    - `(d0)[s0] -> (d0 + s0)` defines **layout mapping**.

#### **(2) Mix of Abstractions**

- MLIR allows affine loops to mix:
    - **Typed SSA values**.
    - **Traditional compiler optimizations (e.g., constant folding, vectorization).**
    - **Polyhedral transformations.**

#### **(3) Smaller Representation Gap**

- Unlike traditional polyhedral compilers that require **lifting code to a drastically different representation**, MLIR retains the original loop structure, reducing transformation overhead.

#### **(4) Faster Compilation Speed**

- Traditional polyhedral compilers rely on **integer linear programming (ILP) and polyhedron scanning**, which can be computationally expensive.
- MLIR **avoids polyhedral scanning**, preserving loop structure directly in IR, reducing computational cost.

---

## **📌 MLIR Affine Dialect Example**

MLIR’s Affine dialect can represent **polynomial multiplication kernels**, as shown below:

```mlir
// Affine loops are Ops with regions.
affine.for %arg0 = 0 to %N {
  affine.for %arg1 = 0 to %N {
    %0 = affine.load %A[%arg0] : memref<? x f32>
    %1 = affine.load %B[%arg1] : memref<? x f32>
    %2 = mulf %0, %1 : f32
    %3 = affine.load %C[%arg0 + %arg1] : memref<? x f32>
    %4 = addf %3, %2 : f32
    affine.store %4, %C[%arg0 + %arg1] : memref<? x f32>
  }
}
```

---

## **📌 Summary**

✅ The Affine dialect offers **structured loops and memory access modeling**, making it suitable for **polyhedral compilation**.

✅ **Compared to traditional polyhedral compilers, MLIR retains higher-level semantics, reducing transformation overhead.**

✅ **Enhances compilation speed, making it ideal for modern machine learning and HPC compilers.** 🚀

# **📌 3 Fortran IR (FIR) - The Intermediate Representation for Fortran**

**Fortran** is a language designed for **numerical computing and high-performance computing (HPC)**. The **LLVM Fortran frontend, "Flang," is currently being developed under the leadership of NVIDIA/PGI**.

Similar to **Swift and Rust, Flang requires a dedicated Intermediate Representation (IR)** to support **Fortran-specific high-performance optimizations**, which are difficult to achieve directly in **LLVM IR**. Consequently, Flang **leverages MLIR to construct a dedicated Fortran IR (FIR, Fortran Intermediate Representation)** to enhance **optimization capabilities and scalability** 🚀.

---

## **📍 Why Does Fortran Need FIR?**

Although LLVM IR is already a powerful intermediate representation, **Fortran has unique characteristics that make efficient optimizations challenging in LLVM IR**, such as:

1. **Advanced Loop Optimizations**
    - Fortran frequently handles large-scale matrix and vector computations in HPC, requiring **special loop optimizations** (e.g., loop fusion, loop tiling, loop interchange, parallelization).
    - FIR provides a **higher-level structured IR**, making these optimizations easier to implement.
2. **Array Copy Elimination**
    - Fortran’s **array semantics differ from pointer-based semantics**, necessitating specific optimizations to **eliminate unnecessary array copies (Copy Propagation)**.
3. **Call Specialization**
    - Fortran allows dynamic procedure selection, but if the target function can be determined at compile time, **function inlining or other specialized optimizations** can improve performance.
4. **Devirtualization**
    - **Fortran’s polymorphism and dynamic function dispatch are not easily represented in LLVM IR**. FIR, however, **treats virtual dispatch tables as a first-class concept**, allowing for finer-grained optimizations.

---

## **📍 FIR’s Virtual Function Dispatch**

FIR provides a **built-in mechanism for dynamic function dispatch**, as shown below:

```mlir
// Define the dispatch table for type(u)
fir.dispatch_table @dtable_type_u {
  fir.dt_entry "method", @u_method
}

// Define a function
func @some_func() {
  // Allocate memory for a variable of type<u>
  %uv = fir.alloca !fir.type<u> : !fir.ref<!fir.type<u>>

  // Dynamically invoke the "method" function
  fir.dispatch "method"(%uv) : (!fir.ref<!fir.type<u>>) -> ()

  // ...
}
```

### **🔍 Breakdown**

1. **`fir.dispatch_table` defines the function dispatch table for Fortran type `type(u)`**
    - **`fir.dt_entry "method", @u_method`** → Indicates that the `method` function points to `@u_method`.
    - This is similar to a **C++ vtable (virtual function table)**, but **FIR integrates it directly into the IR rather than handling it implicitly**.
2. **`fir.alloca` allocates memory for a variable of `type<u>`**
    - This means that within `some_func`, we have a variable `%uv` of `type<u>`.
3. **`fir.dispatch` handles dynamic invocation of "method"**
    - **Compared to LLVM IR, where virtual dispatch requires manually managing a vtable, FIR makes it a first-class concept.**
    - **This design enables better inlining and devirtualization optimizations!**

> 🚀 By structuring virtual function dispatch as a first-class concept, FIR makes Fortran’s polymorphic function calls more intuitive and facilitates advanced optimizations.

---

## **📍 Advantages of MLIR and FIR**

1. **No Need to Reinvent the Wheel (Reusability & Extensibility)**
    - **If Flang were to develop a custom IR from scratch, it would require additional infrastructure**, such as:
        - **Type system**
        - **Optimization passes**
        - **IR transformation mechanisms**
    - **MLIR already provides these foundational tools, allowing Flang developers to focus on Fortran-specific optimizations!**
2. **Reuse of Other Dialects (Dialect Reusability)**
    - **FIR can leverage existing MLIR dialects, such as:**
        - **OpenMP Dialect** → Enables Fortran and C to share the same OpenMP compilation path.
        - **GPU Dialects (e.g., OpenACC)** → Facilitates Fortran’s adaptation to heterogeneous computing.

---

## **📌 Summary**

✅ **Fortran IR (FIR) leverages MLIR to provide high-performance optimization capabilities while avoiding LLVM IR’s limitations.**

✅ **FIR supports virtual dispatch tables, making devirtualization optimizations easier.**

✅ **MLIR allows FIR to integrate with OpenMP, OpenACC, and other dialects, improving Fortran’s extensibility in domain-specific compilers.**

# **📌 4 Domain-Specific Compilers**

Beyond large-scale compiler infrastructures (such as TensorFlow, Fortran, and Polyhedral Code Generation), **MLIR is also used to develop "Domain-Specific Compilers (DSCs)," tailored for specific applications**. This modular and reusable approach reduces development costs 💡.

Here are two case studies demonstrating MLIR’s impact on domain-specific compilers:

1. **Optimizing MLIR Pattern Rewriting**
2. **Lattice Regression Compiler**

---

## **📍 1️⃣ Optimizing MLIR Pattern Rewriting**

### **💡 What is MLIR Pattern Rewriting?**

In **MLIR**, various compiler phases (such as **optimization and lowering**) require pattern matching and transformation, for example:

- Algebraic simplification: `x + 0 → x`
- Instruction selection: Converting high-level operations into lower-level target-specific instructions
- Memory access optimizations

**MLIR provides a flexible "Pattern Rewriting System," allowing developers to define transformation rules** for automatic IR modifications ✨.

---

### **🔍 Traditional MLIR Pattern Rewriting**

Originally, MLIR’s pattern rewriting system was **statically declared**, meaning:

- **All transformation rules were determined at compile time.**
- **Runtime extensibility was not supported.**

This approach was fine for general use cases, but **hardware vendors (e.g., GPU and AI accelerator manufacturers) needed runtime extensibility** to adapt optimizations dynamically 🚀.

---

### **🛠️ Solution: Modeling Pattern Rewriting in MLIR**

To address this, the MLIR team **modeled pattern rewriting as an MLIR dialect**, utilizing **Finite State Machine (FSM) optimization techniques** to generate efficient pattern matchers dynamically.

### **Benefits:**

1. **Dynamic Extensibility**
    - **Vendors can add new transformation rules directly in drivers without recompiling the entire compiler.**
2. **Improved Efficiency**
    - FSM-based instruction selection improves performance, similar to **LLVM SelectionDAG and GlobalISel**.

> ✅ This innovation makes MLIR a strong choice for dynamically extensible AI and GPU compilation pipelines!

---

## **📍 2️⃣ Lattice Regression Compiler**

### **💡 What is Lattice Regression?**

**Lattice Regression** is a machine learning technique commonly used for:

- **Fast inference**
- **Model interpretability**

Applications include:

- **Recommendation systems**
- **Real-time decision-making**

Due to the massive scale of these applications, **performance optimization is critical** 🚀.

---

### **🛠️ Implementing a New Lattice Regression Compiler with MLIR**

Researchers built a **new lattice regression compiler using MLIR**, applying a **search-based optimization approach**.

### **Results:**

- The new compiler was completed in just **three person-months**.
- **Achieved an 8× performance improvement over C++ template implementations.**
- **Improved interpretability of model compilation.**

> ✅ This demonstrates MLIR’s viability for domain-specific compilers, particularly in AI and machine learning!

---

## **📌 Summary**

✅ **MLIR’s pattern rewriting system enables dynamic extensibility for hardware vendors.**

✅ **MLIR-based lattice regression compilers achieved significant performance gains.**

✅ **These examples showcase MLIR’s flexibility in domain-specific compilers!** 🚀

