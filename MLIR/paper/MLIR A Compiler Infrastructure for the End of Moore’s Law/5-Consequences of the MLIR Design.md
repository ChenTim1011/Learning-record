
# **📌 Consequences of the MLIR Design**  

The design of MLIR allows compiler developers to **easily model new languages and compilation abstractions** while **reusing existing general-purpose components**, such as common IR, compilation methods, and optimization passes. This introduces many new opportunities, challenges, and insights 💡.  

One of the most critical design changes is:  

> The common approach to solving compilation problems is to **add new operations (Ops) and types (Types)**, or even **create new dialects (Dialects)** to express higher-level semantics.  

This design philosophy differs from traditional compiler engineering and impacts many aspects. In the following, we will explore how **MLIR’s design affects the reusability of compiler passes (Reusable Compiler Passes).**  

---  

## **📍 1 Reusable Compiler Passes**  

MLIR can represent multiple levels of abstraction simultaneously, which makes developers **want passes to operate across different levels of abstraction**. This leads to a common problem:  

> When MLIR allows the extension of operations (Ops) and types (Types), how can we write a general-purpose compiler pass that works across different dialects (Dialects)?  

In theory, a compiler pass can adopt a **conservatively correct approach** when handling unknown operations, but this may lead to performance degradation. Therefore, we want passes to be **not only correct but also effective in improving program performance** 🚀.  

During the development of MLIR, we identified **four main methods to achieve reusable passes**:  

---  

### **1️⃣ Fundamental Operation Traits**  

Some passes are the **bread and butter of a compiler**, such as:  

- **Dead Code Elimination (DCE)**  
- **Common Subexpression Elimination (CSE)**  

These passes rely on fundamental traits of operations (Ops), such as:  

- **Has No Side Effect**  
- **Is Commutative**  

In MLIR, these traits can be defined using **Op Traits**. For example, in **ODS (Operation Definition Specification)**, we can specify the `Commutative` trait, allowing the CSE pass to recognize that this operation can be reordered.  

### **🔹 MLIR provides structural information, such as:**  

- Whether an operation is a **control flow terminator**  
- Whether an operation is **a region owner and isolated from the outer scope (Isolated-From-Above)**  

This information allows **compiler passes to handle functions, closures, and modules in a general way**, improving pass reusability!  

---  

### **2️⃣ Privileged Operation Hooks**  

Some traits **cannot be represented by a single bit** and require additional **C++ implementation logic**, such as:  

- **Constant Folding**  
- **Algebraic Simplification**  

MLIR provides built-in **Hooks** to support these functionalities. For example, the `getCanonicalizationPatterns` hook allows developers to:  

- Define operation-specific rules, such as `x - x → 0`, `min(x, y, y) → min(x, y)`  
- Apply these rules across **all dialects**  
- Enable **open extensibility** for passes  

This design allows **different dialects to share the same Canonicalization Pass**, replacing many complex passes in LLVM, such as:  

- **InstCombine** (Instruction combination in LLVM IR)  
- **DAGCombine** (Optimization in SelectionDAG)  
- **PeepholeOptimizer** (Simple peephole optimizations)  
- **SILCombine** (Optimization in Swift Intermediate Language)  

These passes in LLVM **are a maintenance and extension burden**, but MLIR’s **open extensibility design** makes development and maintenance significantly easier!  

---  

### **3️⃣ Optimization Interfaces**  

MLIR **not only allows the extension of operations and types but also enables the extensibility of optimization passes**. Some passes require:  

- **Querying specific properties of operations** (e.g., the Inlining Pass needs to determine which functions can be inlined)  
- **Implementing custom cost models**  

To address this, MLIR provides **Optimization Interfaces**, allowing different dialects to extend passes.  

### **🔹 Inlining Pass**  

Inlining is a critical compiler optimization, but different languages have different function call semantics. For example:  

- TensorFlow Graphs may represent functions using **TensorFlow Ops**  
- Flang (Fortran) may use **Fortran Functions**  
- Functional languages may use **Closures**  

However, the inlining pass itself **does not know “what constitutes a function” or “how to handle terminators after inlining”**!  

**Solution: Dialect Inliner Interface**  

```cpp
struct DialectInlinerInterface {
  /// Determines whether an op can be inlined into a region
  bool isLegalToInline(Operation *, Region *, BlockAndValueMapping &) const;

  /// Handles terminator operations after inlining
  void handleTerminator(Operation *, ArrayRef<Value *>) const;
};
```

This design allows:  

- The inlining pass to work across **any dialect** without requiring separate inlining logic for each language  
- Dialect developers to **optionally implement this interface**; if not implemented, the inlining pass will **conservatively handle the operation**  

This approach makes passes **more modular and reduces the complexity of the core compiler**! 🎯  

---  

### **4️⃣ Dialect-Specific Passes**  

Although MLIR encourages general-purpose passes, in some cases, **dialect-specific passes** are still necessary, such as:  

- **Machine instruction scheduling**  
- **Hardware-specific optimizations**  

These passes are mainly used for:  

- **Custom machine code transformations**, such as **optimizations for RISC-V or GPU cores**  
- **Specialized architecture needs**, such as instruction fusion for AI accelerators  

Such passes **do not need to be generalized** but are designed for specific application scenarios, and MLIR **still allows their existence**!  

---  

## **📌 Summary**  

✅ **MLIR improves pass reusability through “Fundamental Operation Traits,” “Privileged Hooks,” “Optimization Interfaces,” and “Dialect-Specific Passes.”**  

✅ **MLIR makes passes easier to extend and maintain, avoiding the high maintenance cost of excessive specialized passes in LLVM.**  

✅ **This design makes MLIR a “scalable, modular” compilation framework, suitable for multiple languages and target architectures!** 🚀  

---

# **📌 2 Mixing Dialects Together**  

One of MLIR’s most revolutionary yet most challenging concepts to understand is its **ability and encouragement to mix operations (Ops) from different dialects (Dialects) within the same program** 💡.  

### **🔹 Why mix different dialects?**  

Some scenarios are intuitive and easy to understand, such as:  

- **Placing computations for both CPU (Host) and AI accelerators (Accelerator) in the same module**  

But **a more interesting and powerful application** emerges when dialects can **directly interoperate**, unlocking an **unprecedented way to reuse compilation infrastructure**! 🚀  

---  

### **📍 Example: Affine Dialect**  

In **Section 5.2**, we used the Affine Dialect to represent **affine control flow and affine mappings**, but the key insight is:  

> The Affine Dialect itself does not depend on the semantics of the operations (Ops) within it.  

This means:  

- The Affine Dialect **can encapsulate operations from different dialects**  
- This design allows **affine transformations to apply to various compute units**  

### **🔹 Concrete Example: Mixing Different Mathematical Computation Dialects**  

We can mix the **Affine Dialect** with **various mathematical computation dialects**:  

1. **Standard Arithmetic Dialect**  
   - A basic arithmetic dialect similar to LLVM IR, providing fundamental operations (`add`, `mul`, `div`, etc.)  
2. **Target-Specific Dialects**  
   - Such as specialized arithmetic instructions for AI accelerators  

This enables:  

- **Loop transformations (e.g., Loop Tiling, Loop Fusion) using the Affine Dialect**  
- **Applicability to different mathematical operations across dialects**  

---

### **📍 Broader Applications: Reusing the OpenMP Dialect**  

MLIR enables the reuse of **generic parallelism (Parallelism) models**, such as:  

- The OpenMP Dialect can be **shared across multiple language IRs**  
- Allowing **C/C++, Fortran, and MLIR frontends to reuse the same OpenMP pass**  

This eliminates the need to **implement separate OpenMP compilation passes for different languages**, instead unifying them under **a single OpenMP Dialect** 🎯.

### **📌 3 Interoperability**

MLIR must interoperate with a **wide range of existing systems**, such as:

- **Machine Learning Graphs** (e.g., TensorFlow Graphs, using protobuf encoding)
- **LLVM IR**
- **Proprietary ISAs**

### **🔹 Problem: Imperfect IR Designs in Different Systems**

Many existing IRs were designed based on **technology needs and constraints at the time**, which may result in:

- **Suboptimal structures**
- **Legacy limitations**

MLIR provides a **more expressive representation** that can **improve these issues with existing IRs**! 💡

---

### **📍 Solution: Create Corresponding Dialects for External Systems**

To simplify integration with these heterogeneous systems, the solution is:

> Create a corresponding dialect for the system and allow for **lossless round-trip conversion**.

This approach:

1. **Simplifies import/export logic**, ensuring the conversion process is **predictable** and **easy to test**.
2. **Uses MLIR for further IR raising or lowering**.

---

### **📍 Specific Examples**

MLIR already has several **dialects specifically designed for interoperability**, including:

### **🔹 1. LLVM Dialect**

- **Maps LLVM IR to MLIR**
- This allows:
  - **Seamless integration with LLVM compiler infrastructure**
  - **Optimizations on front-end IR using MLIR**
  - **Easy access to the LLVM toolchain for developers**

### **🔹 2. TensorFlow Dialect**

- Operations in the TensorFlow graph (like `Switch`, `Merge`, etc.) are **difficult to analyze and convert** in TensorFlow IR.
- In MLIR, these operations can be **elevated** to a higher-level IR using the **TensorFlow Dialect**, which makes:
  - **Optimization easier**
  - **Improved interoperability across different ML systems**

### **🔹 3. Functional Control Flow Dialect**

- In machine learning graphs (ML Graphs), **functional while** and **functional if** constructs are common.
- In traditional IRs (like LLVM IR), these are usually represented as **function calls**, but MLIR can:
  - **Elevate them to regions**, clarifying their semantics.
  - **Lower the cost of analysis and optimization**.

---

### **📌 MLIR Import/Export Benefits from This Design**

Since **importers/exporters are often challenging (e.g., handling binary formats)**, MLIR provides:

- **MLIR Dialects as intermediaries**
- **Unified testing infrastructure for all imports/exports**
- **Reduced complexity and maintenance costs for import/export** ✅

---

### **📌 Summary**

✅ **MLIR supports mixing different dialects, enhancing flexibility in reusing compilation infrastructure**.

✅ **Affine Dialect can mix with other mathematical operation dialects for more powerful optimizations**.

✅ **OpenMP Dialect can be shared across multiple language IRs, increasing reusability**.

✅ **MLIR simplifies import/export and improves interoperability by defining "external system dialects"**.

✅ **This design makes MLIR a more general and flexible IR infrastructure, suitable for various languages and compilation pipelines!** 🚀 - Unopinionated design provides new challenges

## **📌 4 Unopinionated Design Provides New Challenges**

MLIR **allows users to freely define almost any abstract concept**, but it **does not impose specific standards**. This flexibility brings **challenges and opportunities** 🚀.

---

### **🔹 Challenge: Lack of Clear Design Guidelines**

In MLIR, we **can define new dialects, operations (Ops), types**, etc., but:

- **There are no standard guidelines** telling us what is the "best design" and what might lead to issues.
- Many developers are accustomed to **existing IR designs (like LLVM IR, TensorFlow Graph)** but have **limited experience designing IR abstractions**.
- **The art of Compiler IR Design** is still immature, and both academia and industry are still exploring how to define good abstractions.

As a result, many developers might:

- Design IRs that are hard to optimize.
- Make IRs overly complex or redundant.
- Make it difficult for IRs to interoperate with other dialects.

---

### **🔹 Opportunity: A New Research Area**

Despite the challenges, MLIR also gives us the opportunity to:

- **Explore and learn** what kinds of IR designs are more effective.
- **Accumulate experience** and develop best practices.
- **Expand the field of compiler research**, such as:
  - How to design more modular IRs?
  - How to make IRs more optimizable?
  - How to improve interoperability between IRs from different domains (e.g., machine learning, parallel computing)?

The MLIR community is continuously accumulating knowledge in this area, and it could become an **emerging research field in compiler design** 📚.

---

## **📌 5 Looking Forward**

MLIR **differs significantly from traditional compilation infrastructures (like LLVM)**, and even though it has been applied in many systems, its design is still **evolving** 🏗️.

### **🔹 Many Unknowns Ahead**

Although MLIR's design has been successful, there are still many **unresolved design points** that need more time and research to explore, including:

1. **Increasing Out-of-Tree Dialects**
   - Out-of-Tree Dialects refer to dialects that are **not directly merged into the main MLIR codebase**.
   - The **growth and management** of these dialects will become a new challenge.
   
2. **More Frontends Using MLIR**
   - More languages are choosing to use MLIR as their IR (e.g., Swift, Julia, TensorFlow).
   - How can these frontends better utilize MLIR's capabilities?

3. **Possible Applications to Abstract Syntax Trees (ASTs)**
   - MLIR currently focuses on intermediate representation (IR), but can it **expand to the AST level**?
   - This could blur the boundary between frontend and intermediate layers, opening new possibilities.

4. **Applications in Structured Data**
   - Structured data formats like **JSON** and **Protocol Buffers** might also benefit from optimization using MLIR.
   - This could be a new opportunity for **data processing systems** (e.g., databases, machine learning frameworks).

---

## **📌 Summary**

✅ **MLIR allows highly flexible design but lacks specific "best practices," making it both a challenge and an opportunity.**

✅ **The art of IR design is still evolving, and the MLIR community is accumulating experience, which may become a new research direction.**

✅ **In the next few years, MLIR still holds many unsolved mysteries in areas like Out-of-Tree Dialects, more frontend languages, AST applications, and structured data processing—worth exploring further.** 🚀