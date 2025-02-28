## **Summary**

MLIR is a flexible, scalable intermediate representation that allows:

1. **Customizable IRs** (like LEGO blocks).
2. **SSA + Regions** for better structure and analysis.
3. **Gradual lowering** to avoid losing meaning too early.
4. **Retaining high-level semantics** until necessary.
5. **IR validation** to ensure correctness.
6. **Declarative rules** for better readability and maintenance.
7. **Source tracking and traceability** for transparency.

## 1. **Little Builtin, Everything Customizable**

- MLIR provides basic components (like **types, operations, and attributes**), but everything else is customizable.
- **Analogy**: Like "LEGO blocks" – MLIR offers basic pieces, and users can combine them freely to build various structures (unlike LLVM IR, which is more specific to CPUs and C-like languages).
- **Benefits**: It supports different scenarios, including **machine learning models**, **abstract syntax trees (ASTs)**, **numerical optimizations**, and **traditional compiler IRs**.

---

## 2. **SSA and Regions**

- MLIR uses **SSA (Static Single Assignment)** and allows **nested regions** for a more structured IR, unlike flat control flow graphs (CFGs).
- **Analogy**: 
    - **SSA**: Like Excel cells where each can only be set once, making data flow clearer.
    - **Regions**: Like chapters and sub-chapters in a document, making complex structures easier to represent.
- **Why**: 
    - **Optimization**: SSA ensures variables don't change unexpectedly, aiding analysis.
    - **Higher-level Structures**: Easier representation of things like **concurrency** and **closures**.

---

## 3. **Progressive Lowering**

- MLIR allows gradual transformation from high-level to low-level representations instead of directly producing machine code.
- **Analogy**: Like writing a paper – start with an outline (high-level) and add details gradually (low-level).
- **Why**: 
    - **Flexibility**: Different hardware requires different IRs, enabling more flexible optimizations.
    - **Error Reduction**: Step-by-step conversion ensures each stage remains readable and verifiable.

---

## 4. **Maintain Higher-Level Semantics**

- MLIR retains high-level program information until it's necessary to lower it to machine code.
- **Analogy**: Like translating a novel, keeping the story structure intact before focusing on specific details.
- **Benefits**: 
    - **Better Optimizations**: Retaining high-level semantics helps with optimizations, like keeping loop structures intact.
    - **Easier Debugging**: Retaining meaning through the compilation process makes debugging easier.

---

## 5. **IR Validation**

- MLIR allows validation of IR to ensure the compilation process is error-free.
- **Analogy**: Like quality control in manufacturing – ensuring each part meets standards before final assembly.
- **Why**: 
    - **Scalability**: Ensures custom IR extensions follow rules.
    - **Debugging**: Helps quickly identify errors in IR transformations.

---

## 6. **Declarative Rewrite Patterns**

- MLIR enables developers to define IR transformations using **declarative syntax** rather than writing manual code.
- **Analogy**: Like using a "rule table" for syntax transformation instead of writing lots of if-else logic.
- **Why**: 
    - **Readability & Maintainability**: Declarative rules are clearer and easier to understand than manual transformation code.
    - **Better Analysis**: Machine-readable rules enable automatic correctness and efficiency checks.

---

## 7. **Source Location Tracking & Traceability**

- MLIR tracks the source of each operation, enabling developers to trace how the code was transformed.
- **Analogy**: Like version control for documents, allowing you to track changes and ensure transparency.
- **Applications**: 
    - **Safety-critical applications** (e.g., medical devices, avionics) to ensure optimizations don't affect safety.
    - **Privacy-sensitive applications** (e.g., cryptography) to ensure important security mechanisms aren’t removed by the compiler.

---



# **Design Principle - Little Builtin, Everything Customizable**
:

👉 **Minimal built-in components, everything is customizable.**

MLIR adopts a minimalist core concept, providing only basic types, operations, and attributes. This allows users to freely extend the IR without being constrained by specific semantics or architectures. Such a design makes MLIR adaptable to various needs, making it suitable for a wide range of applications, including machine learning, mathematical computations, abstract syntax trees (ASTs), and instruction-level IRs like LLVM IR.

However, this high degree of customizability also introduces the risk of **internal fragmentation**, where different user-defined extensions may be incompatible, making the system harder to maintain.

---

## **Terminology Explained**

This discussion involves several key concepts. Let's first explain these terms before diving deeper into the meaning of the statement.

### 1️⃣ **Intermediate Representation (IR)**

- IR is a program representation between high-level languages (like C/C++) and low-level machine code (like x86 instructions), commonly used inside compilers.
- For example, LLVM IR is the IR used by the LLVM compiler. It provides an SSA-based instruction set, allowing optimizations and transformations.

### 2️⃣ **Types**

- In MLIR, `Type` represents the data type of a variable or value, such as:
  - `f32` (32-bit floating point)
  - `i64` (64-bit integer)
  - `tensor<4xf32>` (a tensor of four `f32` values)
- MLIR allows developers to define custom types like `TensorType`, `StructType`, or specialized types for machine learning models, such as `QuantizedType`.

### 3️⃣ **Operations (Ops)**

- In MLIR, every instruction is an **Operation** (Op), similar to LLVM IR instructions but more generic.
- Examples:
  - `add %x, %y : f32` (floating-point addition)
  - `toy.transpose %A : (tensor<2x3xf32>) -> tensor<3x2xf32>` (Toy dialect matrix transposition)
- Users can define their own operations, such as `CustomOp`, to support new application domains.

### 4️⃣ **Attributes**

- Attributes store metadata in MLIR, similar to constants in LLVM IR.
- Examples:
  - `DenseElementsAttr`: Stores constant values of a tensor
  - `StringAttr`: Stores string labels
  - `IntegerAttr`: Stores integer constants
- Attributes are often used alongside operations to provide additional compile-time information.

### 5️⃣ **AST (Abstract Syntax Tree)**

- An AST is a tree-like representation of a program’s syntactic structure, mainly used in the **frontend** of a compiler for parsing.
- Example AST for `a = b + c;`:

  ```
       =
      /  \
     a    +
         /  \
        b    c
  ```

- **Difference between IR and AST:**
  - AST is **closer to the original syntax** and useful for semantic analysis.
  - IR is **more focused on computation** and is suitable for optimization.

### 6️⃣ **Polyhedral Model**

- The polyhedral model is a mathematical approach often used for analyzing and optimizing **loop computations**, particularly in high-performance computing (HPC) and machine learning.
- In MLIR, polyhedral techniques help **identify parallelism** and facilitate loop optimizations like loop tiling and loop fusion.

### 7️⃣ **Control Flow Graph (CFG)**

- CFG is the core structure of traditional IRs (like LLVM IR), representing program control flow using **basic blocks (BBs)** and jumps (`br`, `switch`).
- Example:

  ```
  BB1:   %x = add %a, %b
         br BB2
  BB2:   %y = mul %x, %c
         ret %y
  ```

- CFG is effective for traditional compiler optimizations but may lack flexibility for high-level control structures like `if` statements, loops, and function closures.

### 8️⃣ **Fragmentation**

- **Refers to the incompatibility between different system components, increasing development complexity.**
- MLIR allows developers to define custom ops and types, leading to high flexibility. However, if different teams independently create their own IR extensions, it may result in:
  - Incompatible **dialects**
  - **Non-interoperable IRs**
  - Difficult-to-maintain **toolchains**

---

## **Breaking Down the Statement**

Now, let’s analyze the statement in detail:

### **1. "The system is based on a minimal number of fundamental concepts, leaving most of the intermediate representation fully customizable."**

👉 **MLIR provides only the most basic core concepts, leaving the rest fully customizable.**

- **Core concepts:** Types, Operations, Attributes
- **Customizable parts:** Syntax, operations, optimization techniques, etc.

---

### **2. "A handful of abstractions—types, operations, and attributes, which are the most common in IRs—should be used to express everything else, allowing fewer and more consistent abstractions that are easy to comprehend, extend, and adopt."**

👉 **MLIR expresses everything through a small number of abstractions (Types, Ops, Attributes), making the IR more unified and extensible.**

- **Advantages:**
  1. **Simplicity** – fewer concepts to learn
  2. **Extensibility** – custom IRs for different domains
  3. **Consistency** – different applications share a common base structure

---

### **3. "A success criterion for customization is the possibility to express a diverse set of abstractions including machine learning graphs, ASTs, mathematical abstractions such as polyhedral, Control Flow Graphs (CFGs), and instruction-level IRs such as LLVM IR, all without hard coding concepts from these abstractions into the system."**

👉 **MLIR’s success is measured by its ability to represent various abstractions without hardcoding them.**

- MLIR aims to support:
  - **Machine Learning Graphs**
  - **Abstract Syntax Trees (ASTs)**
  - **Mathematical models (Polyhedral)**
  - **Traditional Control Flow Graphs (CFGs)**
  - **LLVM IR**

---

### **4. "Certainly, customizability creates a risk of internal fragmentation due to poorly compatible abstractions."**

👉 **High customizability may lead to internal fragmentation, as different extensions might not be compatible.**

- For example, if different teams define IRs without standardization, interoperability becomes a challenge.

---

### **5. "The system should encourage one to design reusable abstractions and assume they will be used outside of their initial scope."**

👉 **MLIR should encourage developers to design reusable abstractions, not just for specific applications.**

- This helps reduce fragmentation, making it easier for different dialects to interoperate and integrate.

---

## **Conclusion**

MLIR follows a **minimal core + high extensibility** design, allowing developers to create IRs for various domains. However, it must also mitigate fragmentation risks to ensure a stable and interoperable ecosystem.


---

# **Design Principle - SSA**

This discussion focuses on **Static Single Assignment (SSA) form** and **Regions** in compiler Intermediate Representations (IRs). It explores how to support high-level abstractions (such as nested loops) while maintaining efficient analysis and transformation capabilities.  

There are many key concepts here, so let's break them down first before analyzing the meaning of this passage.  

---

### **Terminology Explanation**  

1. **SSA (Static Single Assignment)**  
   - SSA is a common representation in compiler IRs, where each variable can be **assigned only once**. The value of any variable must come from a previously defined variable or a `phi` function (used for selecting values from different basic blocks).  
   - The advantage of SSA is that it simplifies **dataflow analysis**, making optimizations easier.  
   - LLVM IR adopts the SSA form, meaning that all variables in LLVM can only be assigned once. If a value needs to be updated, a new variable must be created.  

2. **Region**  
   - A **Region** is a high-level control flow structure introduced by **MLIR (Multi-Level IR)**, differing from traditional Control Flow Graphs (CFGs).  
   - In linearized IRs like LLVM IR, control flow is represented using **Basic Blocks** and `br` (branch instructions). In contrast, Regions allow the compiler to directly express **nested structures** such as `if` and `for` statements.  
   - Regions make IR more similar to high-level programming languages like C/C++ or Python, where functions, loops, and conditional statements are structured instead of being broken down into `br` instructions and basic blocks.  

3. **Control Flow Graph (CFG)**  
   - Traditional IRs (such as LLVM IR) primarily use **Control Flow Graphs (CFGs)** to represent program execution flow.  
   - A CFG consists of **Basic Blocks (BBs)**, each containing a sequence of instructions, with transitions between blocks determined by branches (`br`, `switch`, etc.).  
   - While this representation is well-suited for many compiler optimizations, it is not ideal for preserving **structured control flow** (e.g., `if` and `for` loops). These structures are typically "flattened" into multiple basic blocks.  

4. **Canonicalization**  
   - Canonicalization refers to converting different but semantically equivalent IR representations into a **standard form** to facilitate analysis and optimization.  
   - For example, in LLVM IR, loops are often transformed into a standardized structure consisting of:  
     - **Pre-header**: Performs pre-computations to ensure that the main loop body contains only core computations.  
     - **Header**: The primary control point that determines whether the loop should continue executing.  
     - **Latch**: A block that typically jumps back to the `header`, controlling loop continuation.  
     - **Body**: The main computation logic of the loop.  
   - This standardization helps LLVM apply optimizations consistently across different languages.  

---

### **Sentence-by-Sentence Analysis**  

Now, let's break down the meaning of the passage:  

1. **"The Static Single Assignment (SSA) form is a widely used representation in compiler IRs."**  
   - SSA is a common representation in compiler IRs (such as LLVM IR). Its advantage is that it simplifies dataflow analysis, making the IR structure clearer and optimizations easier.  

2. **"It provides numerous advantages including making dataflow analysis simple and sparse, is widely understood by the compiler community for its relation with continuation-passing style, and is established in major frameworks."**  
   - The advantages of SSA:  
     - **Simplifies dataflow analysis** since each variable is assigned only once, making it easier to track values.  
     - **Related to Continuation-Passing Style (CPS)**, a programming style commonly found in functional languages that influenced IR design.  
     - **Adopted by major compiler frameworks**, such as LLVM and GCC.  

3. **"While many existing IRs use a flat, linearized CFG, representing higher-level abstractions pushes introducing nested regions as a first-class concept in the IR."**  
   - Traditional IRs (such as LLVM IR) primarily use **linearized CFGs**, where control flow is expressed through basic blocks and `br` instructions.  
   - However, to better represent **high-level abstractions** (such as functions, loops, and conditionals), MLIR introduces **Regions as first-class citizens** in the IR, allowing direct support for nested structures.  

4. **"This goes beyond the traditional region formation to lift higher-level abstractions (e.g., loop trees), speeding up the compilation process or extracting instruction, or SIMD parallelism."**  
   - Unlike traditional IRs that break down `for` loops into basic blocks, MLIR allows `for` loops to be represented directly as **nested Regions**. This enables:  
     - **Faster compilation**, as unnecessary CFG transformations are reduced.  
     - **Easier extraction of instruction-level parallelism (SIMD, Single Instruction Multiple Data)**, since the compiler can directly analyze loop structures within a Region.  

5. **"One specific challenge is to make CFG-based analyses and transformations compose over nested regions."**  
   - **Challenge: How to make CFG-based analysis and transformations work with Regions?**  
   - Traditional LLVM optimization techniques are mostly based on CFGs, whereas MLIR’s Region concept operates at a higher level. Therefore, adapting LLVM’s optimization passes for Regions requires additional design considerations.  

6. **"In doing so, we aim to sacrifice the normalization, and sometimes the canonicalization properties of LLVM."**  
   - **MLIR sacrifices some of LLVM’s normalization and canonicalization properties to support Regions.**  
   - LLVM enforces a standardized control flow (such as structured loop transformations), whereas MLIR retains the original high-level structure where possible.  

7. **"By offering such a choice, we depart from the normalization-only orientation of LLVM while retaining the ability to deal with higher-level abstractions when it matters."**  
   - MLIR provides **flexibility**, allowing users to choose whether to represent loops as **nested Regions** or as **linearized CFGs**, unlike LLVM, which strictly enforces CFG-based structures.  

---

### **Summary**  

- **LLVM IR** uses SSA and CFG to represent programs, simplifying dataflow analysis but making it harder to maintain high-level program structure.  
- **MLIR** introduces **Regions**, which allow direct representation of nested structures such as `if` and `for` loops. This improves compilation speed and optimizations.  
- **The main challenge** is adapting traditional CFG-based optimizations to work with Regions while balancing high-level abstraction with standard IR transformations.  


---

### **Design Principle - Progressive Lowering**

This section mainly discusses **MLIR’s support for the mechanism of "Progressive Lowering"**, which allows the program to be lowered step by step from high-level representations (such as AST or high-level IR) to lower-level representations (such as LLVM IR or machine code). The process should be **flexible and extensible**, unlike traditional compilers that have a fixed hierarchical structure.

This design helps solve the **Phase Ordering Problem**, allowing different optimization transformations (such as constant propagation, value numbering, and dead code elimination) to be combined more flexibly without being restricted by the traditional order of compilation passes.

---

### **Glossary**

This section involves several compiler-related terms, so let’s explain them first.

#### **1️⃣ Lowering**

- **Lowering** refers to the process of gradually converting a high-level IR into a lower-level IR until the final machine code is generated.
- For example:
  - **AST → High-level IR (MLIR Toy Dialect)**
  - **High-level IR → Low-level IR (LLVM Dialect)**
  - **LLVM IR → Machine IR**
  - **Machine IR → Machine Code (x86, ARM, etc.)**
- **Progressive Lowering** means that this process occurs **in stages** rather than a one-time transformation.

---

#### **2️⃣ Abstraction Levels**

- **Different representations in a compiler correspond to different levels of abstraction:**
  1. **High-level Representation**: 
     - For example, **AST (Abstract Syntax Tree)**, which is closer to the original structure of the code.
  2. **Mid-level Representation**: 
     - For example, **MLIR high-level dialects (like Toy Dialect)** or **LLVM IR**.
  3. **Low-level Representation**: 
     - For example, **SelectionDAG (LLVM’s representation for instruction selection)** or **MachineInstr (Machine-level IR)**.
  4. **Lowest-level Representation**: 
     - For example, **MCInst (LLVM MC layer, machine code-level representation)**.

---

#### **3️⃣ Open64 WHIRL Representation**

- **Open64** is an open-source compiler infrastructure primarily used for Fortran and C/C++, and it features a **WHIRL** IR with **five different levels of IR**:
  - **Very High** → Close to AST.
  - **High** → High-level optimizations.
  - **Mid** → Conversion to more generic instructions.
  - **Low** → Close to LLVM IR.
  - **Very Low** → Close to machine code.
- This design **fixes five transformation layers**, which are more rigid and lack extensibility.

---

#### **4️⃣ Clang Compilation Process**

When Clang compiles C/C++, it undergoes **several fixed lowering stages**:

1. **AST (Abstract Syntax Tree)**:
   - The frontend parses the source code to build the syntax tree.
2. **LLVM IR**:
   - Converted to LLVM’s intermediate representation for optimization.
3. **SelectionDAG**:
   - Converted to **Selection DAG** (used for instruction selection).
4. **MachineInstr (Machine-level IR)**:
   - Converted to machine instructions specific to the hardware architecture.
5. **MCInst (Machine Code-level IR)**:
   - Finally converted to binary machine code.

**Problem:** This method is too rigid and difficult to extend, making it less adaptable to new needs, such as support for GPUs, AI accelerators, etc.

---

#### **5️⃣ Phase Ordering Problem**

- **The order of optimizations can affect the final optimization result**.
- For example:
  - If we first perform **Value Numbering** and then **Constant Propagation**, we may discover more opportunities for simplification.
  - If the order is reversed, some optimization opportunities might be missed.
- **Traditional compilers often fix certain optimization orders, but this might not be the best choice.**

---

#### **6️⃣ Compiler Passes**

- A **pass** refers to an optimization or transformation stage in the compiler.
- The four types of passes mentioned here:
  1. **Optimizing Transformations**: 
     - For example, **Loop Unrolling**, **Constant Propagation**.
  2. **Enabling Transformations**: 
     - These transformations don't directly improve performance but enable other optimizations, such as **Loop Normalization**.
  3. **Lowering**: 
     - For example, converting from **MLIR → LLVM IR → Machine IR**.
  4. **Cleanup**: 
     - For example, **Dead Code Elimination (DCE)**.

---

### **Core Concepts of This Section**

#### **MLIR's Advantage: Supporting Progressive Lowering**

1. **Unlike Clang or Open64 with fixed IR layers**, MLIR provides more flexible **Dialects**, allowing developers to define their own IR levels.
2. **Solving the Phase Ordering Problem**: 
   - Optimizations can be mixed at the level of individual operations, without relying on a fixed sequence of compilation passes.
3. **Adaptation to Different Hardware Architectures**: 
   - **Traditional compilers are designed for CPUs**, but MLIR’s flexibility allows it to support GPUs, AI accelerators, and other hardware.

---

### **Challenges and Issues**

#### **1. How to Choose the Best Lowering Strategy?**

- If the lowering process is too fragmented, it could affect compilation performance.
- If it’s too coarse, some optimization opportunities might be missed.

#### **2. How to Design the "Appropriate Compiler Pass Order"?**

- **The interaction between different optimizations is complex**—how can MLIR’s pass system automatically find the optimal order?

#### **3. How to Ensure Interoperability Between Different Dialects?**

- For example, if one MLIR dialect **does not fully support** lowering to LLVM IR, can it still communicate with LLVM?

---

### **Summary**

1. **Traditional compilers (Clang, Open64) use fixed IR layers, which limits extensibility**.
2. **MLIR adopts "Progressive Lowering," allowing developers to define their own IR levels and mix optimizations and lowering operations flexibly**.
3. **This design solves the Phase Ordering Problem, enabling the compiler to choose the appropriate lowering strategy based on different needs**.
4. **However, this brings challenges, such as ensuring interoperability between different dialects and choosing the best pass order**.

MLIR’s flexible architecture makes it suitable not only for traditional CPUs but also for GPUs, AI accelerators, and other computational platforms.

---

### **Design Principle - Maintain Higher-level Semantics**

This section focuses on **retaining higher-level semantics** during the compilation process, meaning that the structure and semantics of computations should be preserved as much as possible before lowering the code (to LLVM IR or machine code). Attempting to recover this information after it has been lowered is typically **fragile and error-prone**.

Additionally, MLIR supports **mixed abstraction levels**, allowing some parts to maintain higher-level semantics while other parts are lowered to a lower-level representation. This is particularly important for heterogeneous computing platforms like GPUs and AI accelerators.

---

#### **Glossary and Original Text Comparison**

---

#### **1️⃣ Maintain Higher-level Semantics**

> The system needs to retain higher-level semantics and structure of computations that are required for analysis or optimizing performance.

**The system needs to retain higher-level semantics and structure of computations for analysis or optimization purposes.**

- **Higher-level semantics** refers to the original structure of the program, such as functions, loops, mathematical operations, data flow, etc.
- **If lowering happens too early, these semantics may be lost**, which can impact optimization and analysis.
- **Example**:
  - If we have a `for` loop:
  
```cpp
for (int i = 0; i < N; i++) {
    A[i] = B[i] + C[i];
}
```

  - Directly lowering it to a Control Flow Graph (CFG) might lose the concept of "this is a loop," making loop optimizations difficult.
  - However, if the loop structure is retained, the compiler can apply optimizations like **vectorization** or **unrolling** to improve performance.

---

#### **2️⃣ Attempts to Raise Semantics Once Lowered Are Fragile**

> Attempts to raise semantics once lowered are fragile and shoehorning this information into a low-level often invasive.

**Trying to recover higher-level semantics after lowering is fragile and forcing this information back into the low-level representation often disrupts the entire compilation process.**

- **Why is this not a good approach?**
  - Once high-level structures are converted into low-level IR, some information is completely lost, for example:
    **After lowering to LLVM IR**:

```cpp
x = a + b;
y = x * c;
```

```llvm
%1 = add i32 %a, %b
%2 = mul i32 %1, %c
```

  - At this point, variables `x` and `y` no longer exist, and the concept of blocks might be lost as well.
  - **Trying to recover the original semantics afterward is difficult and potentially impossible.**

- **Why is "shoehorning" semantics back into the low-level invasive?**
  - If we try to forcefully preserve the structure through **debug information** or **annotations**, then:
    - Every pass must understand these extra annotations, affecting the design and correctness of each pass.
    - This makes the compiler harder to extend since different passes must adapt to these semantic annotations.

---

#### **3️⃣ Maintain Structure of Computation and Progressively Lower**

> Instead, the system should maintain structure of computation and progressively lower to the hardware abstraction.

**The system should maintain the structure of computations and progressively lower them to hardware abstraction.**

- **Progressive Lowering**: 
  - Don’t transform everything at once. Instead, **retain higher-level information until it is necessary to lower**.
  - For example, in MLIR loop optimization, we can:
    1. **Keep the LoopOp structure** for unrolling or parallelization optimizations.
    2. Only after optimization is done, lower it to a basic block in LLVM IR.

- **Example**:
  - Suppose we have a MLIR loop dialect:

```mlir
scf.for %i = %c0 to %N step %c1 {
    %val = addf %a, %b
}
```

  - In MLIR, we retain this structure and apply optimizations like **loop unrolling**.
  - Only when we no longer need this structure do we lower it to LLVM IR:

```llvm
br label %loop_header
loop_header:
  %i = phi i32 [0, %entry], [%next, %loop_body]
  %val = fadd float %a, %b
  %next = add i32 %i, 1
  br label %loop_body
```

  - **This ensures that we lower the information only when necessary, and don’t lose structure too early.**

---

#### **4️⃣ Preserving Structured Control Flow**

> For example, the system should preserve structured control flow such as loop structure throughout the relevant transformations.

**The system should preserve structured control flow, such as loop structure, until necessary transformations are completed.**

- **Structured Control Flow**
  - For example, if/while/for structured statements should be retained at the MLIR level, rather than immediately converted to LLVM basic blocks.
  - This allows optimizations (like loop optimizations) to proceed more effectively.
  - **If we prematurely convert to CFG, many high-level optimizations will become impossible.**

---

#### **5️⃣ Mixing Different Levels of Abstractions**

> Mixing different levels of abstractions and different concepts in the same IR is a key property of the system.

**Allowing different levels of abstraction and concepts to coexist in the same IR is a key feature of the system.**

- **Why is this important?**
  - **Heterogeneous computing** requires different levels of IR to coexist:
    - For example, GPU code might maintain high-level matrix operations, while CPU code has already been lowered to scalar instructions.
  - This design allows:
    - **Certain parts to undergo high-level optimizations**.
    - **Other parts to be lowered to low-level IR for machine code generation.**

---

#### **6️⃣ Custom Accelerators**

> This would enable, for instance, a compiler for a custom accelerator to reuse some higher-level structure and abstractions defined by the system.

**For example, this allows compilers for custom accelerators to reuse the high-level structures and abstractions built into the system while supporting specific scalar/vector instructions.**

- **This is particularly important for AI accelerators**:
  - **Certain AI computations** (like matrix multiplication) can retain high-level representation, while others are lowered to vectorized instructions.
  - This enables **different computational units (like CPUs, GPUs

, TPUs)** to coexist without re-compiling every time.

---

#### **Summary**

**MLIR provides a design that allows a gradual lowering process**, while **maintaining the structure of high-level operations** throughout the transformation process. This preserves **semantic integrity** and **flexibility** for hardware targets, making it an ideal solution for diverse, heterogeneous computing platforms.

---

### **Design Principle - IR Validation**

This section focuses on the **Validation Mechanism of IR (Intermediate Representation)**, especially how to ensure the correctness of IR in an **open and extensible compiler ecosystem**. Since MLIR allows users to define custom dialects, a **powerful and easy-to-define validation mechanism** is required to ensure that IR maintains correctness and consistency after extension.

The section also touches upon the **hierarchical structure of IR (Region → Block → Operation)** and the potential development of **Translation Validation** methods to verify the correctness of the compilation process.

---

## **Glossary and Original Text Comparison**

---

### **1️⃣ IR Validation**

> The openness of the ecosystem calls for an extensive validation mechanism.
> 
> **While verification and testing are useful to detect compiler bugs, and to capture IR invariants, the need for robust validation methodologies and tools is amplified in an extensible system.**

**An open ecosystem requires a comprehensive validation mechanism. While "verification" and "testing" can be used to detect compiler bugs and capture IR invariants, robust validation methods and tools become even more critical in an extensible system.**

- **Verification**:
    - Ensures that the IR conforms to **syntax and structural rules**, such as:
        - Does each Operation (Op) have valid input/output types?
        - Does a Block contain a terminator instruction?
        - Do Regions comply with domain-specific rules?
- **Testing**:
    - Executes **test cases** to check if the IR runs correctly, such as:
        - Can it be correctly translated to LLVM IR?
        - Does it conform to the expected semantics?
        - Can it generate the correct machine code?
- **Why is validation more important than testing?**
    - In an **extensible (custom dialect)** IR system like MLIR, each **custom dialect** may have its own rules.
    - **Testing can only cover limited scenarios**, but **validation ensures the structure and semantics of IR are correct**, making it more important.

---

### **2️⃣ Declarative Validation**

> The mechanism should aim to make this easy to define and as declarative as practical, providing a single source of truth.

**The validation mechanism should be designed to be easy to define and as **declarative** as possible, providing a single source of truth.**

- **Declarative Validation**:
    - **Defines a set of rules** for the validation mechanism to automatically check if the IR conforms to these rules, rather than relying on writing extensive validation functions.
    - For example, in MLIR, we can declare with **ODS (Operation Definition Specification)**:

        ```
        def MyOp : Op<MyOp, [NoSideEffect]> {
          let summary = "A custom operation";
          let arguments = (ins I32:$input);
          let results = (outs I32:$output);
        }
        ```

    - MLIR will **automatically check**:
        1. Whether `MyOp` has no side effects (`NoSideEffect`).
        2. Whether its input is `I32`.
        3. Whether its output is `I32`.

- **Single Source of Truth**:
    - Ensures that validation rules are managed centrally, avoiding inconsistencies where different parts of the system maintain their own validation logic.

---

### **3️⃣ MLIR's IR Hierarchy**

> Operations contain a list of regions, regions contain a list of blocks, blocks contain a list of Ops, enabling recursive structures.

**Operations (Ops) contain Regions, Regions contain Blocks, and Blocks contain Ops, forming a recursive structure.**

MLIR’s IR has a hierarchical structure as follows:

1. **Operation (Op)**:
    - The **basic unit** in MLIR, similar to LLVM IR instructions.
    - Each Op can have inputs, outputs, and attributes.
    - Can contain **Regions** to describe control flow further.
2. **Region**:
    - **Ops can have Regions**, similar to function bodies or loop blocks.
    - A Region contains **Blocks** to express control flow.
3. **Block**:
    - A Block contains a series of **Ops**, representing a sequence of executions.
    - Each Block must have a **terminator instruction** to indicate how control flows.
4. **Op (Operation)**:
    - An Op can also contain nested **Regions**, forming a recursive structure.

📌 **Example IR**:

```
%results:2 = "d.operation"(%arg0, %arg1) ({
  // Region 1 (contains multiple Blocks)
  ^block(%argument: !d.type):
    // Block contains Operations
    %value = "nested.operation"() ({
      // Op contains a Region
      "d.op"() : () -> ()
    }) : () -> (!d.other_type)

    // Consume the value
    "consume.value"(%value) : (!d.other_type) -> ()

  ^other_block:
    // Block terminator
    "d.terminator"() [^block(%argument : !d.type)] : () -> ()
})
// Operations can have Attributes
{attribute="value" : !d.type} : () -> (!d.type, !d.other_type)
```

This IR represents:

1. **`d.operation`** as an Operation with a **Region**.
2. **The Region contains multiple Blocks** (`^block`, `^other_block`).
3. A Block contains **multiple Ops**, where `"nested.operation"` also has **another Region** inside.
4. This design allows **recursive structures**, making the IR more expressive.

---

### **4️⃣ Translation Validation**

> A long-term goal would be to reproduce the successes of translation validation and modern approaches to compiler testing.

**A long-term goal is to replicate the successes of **Translation Validation** and modern compiler testing methods.**

- **What is Translation Validation?**
    - **Ensures that the compilation process does not alter the program’s semantics**.
    - Instead of checking whether the output is correct, it checks whether the compiler **faithfully translates** the program.
    - For example:
        - **Frontend AST → MLIR**
        - **MLIR → LLVM IR**
        - **LLVM IR → Machine Code**
    - At each step, **SMT Solvers or equivalence checking** are used to ensure semantic preservation.
- **Current Challenges**:
    - MLIR **allows for custom dialects**, which makes translation validation more complex.
    - Stronger **formal verification** methods are needed to ensure that all lowering processes are correct.

---

## **Summary**

1. **The validation mechanism is particularly important in an open IR system like MLIR** because it allows custom dialects.
2. **MLIR IR has a hierarchical structure (Operations → Regions → Blocks → Ops)**, enabling nested and recursive representations.
3. **Translation Validation is a significant future challenge**, ensuring the compilation process does not alter semantics.
4. **MLIR uses declarative validation**, with ODS defining IR rules to ensure consistency and extensibility.

--- 


- **Design Principle - Declarative Rewrite Patterns**

This section focuses on **Declarative Rewrite Patterns**, discussing how to perform program transformations using **machine-analyzable declarative rules**. This approach enhances **extensibility**, ensures the **correctness** and **reproducibility** of the transformation process, and allows for flexible transformation mechanisms within the compiler, supporting multi-level IR lowering.

The section also highlights some research challenges, such as:
1. **How to design rewrite rules?**
2. **How to manage different levels of rewrite strategies?**
3. **How to ensure monotonicity and predictability during the rewriting process?**

---

## **Terminology and Corresponding Original Text**

---

### **1️⃣ Declarative Rewrite Patterns**

> "Defining representation modifiers should be as simple as that of new abstractions; a compiler infrastructure is only as good as the transformations it supports."

**Defining IR transformation rules should be as simple as defining new abstractions; the value of a compiler infrastructure depends on the transformations it supports.**

Here, **"IR transformations (Representation Modifiers)"** refers to how IR is modified for optimizations or lowering (e.g., converting a high-level MLIR dialect to a lower-level representation or performing optimizations like constant folding). **"Declarative"** means using structured rules to describe rewrites instead of manually writing complex transformation logic.

---

### **2️⃣ Machine-analyzable Format**

> "Common transformations should be implementable as rewrite rules expressed declaratively, in a machine-analyzable format to reason about properties of the rewrites such as complexity and completion."

**Common transformations should be defined as rewrite rules in a declarative, machine-analyzable format, so that properties such as complexity and completeness of the rewrites can be analyzed.**

- **Machine-analyzable format** refers to the format in which transformation rules can be **automatically checked and validated**, allowing for analysis of the **computational complexity** and ensuring that the **transformation process can be fully executed (Completion)**.
- **Why is this important?**
    - Many compiler transformation rules are **manually written**, which can lead to:
        - **Difficulty in analyzing correctness**, potentially resulting in incorrect transformations.
        - **Transformation order might affect the result**, making it difficult to predict.
    - **Using declarative rules + machine analysis** ensures:
        - **Consistency** of the rules to avoid conflicts.
        - **Reproducibility** of the transformation process.
        - **Correctness** of compiler optimizations.

📌 **Example of MLIR Declarative Rewrite:**
``` 
RewritePattern(
  match("arith.addi(%x, 0)"),
  replace("%x")
)
```
This rule indicates:
- If **`arith.addi`** (integer addition) has **0** as its second operand, it directly replaces it with the first operand, effectively removing `x + 0`.

This declarative syntax is easier to analyze and maintain than manually written C++ transformation functions.

---

### **3️⃣ Rewrite Systems**

> "Rewriting systems have been studied extensively for their soundness and efficiency, and applied to numerous compilation problems, from type systems to instruction selection."

**Rewrite systems have been extensively studied for soundness and efficiency, and applied to many compilation-related problems, from type systems to instruction selection.**

- **Rewrite systems** are mathematical models that recursively modify representations using a set of rules.
- This concept is common in **compiler optimization**, such as:
    - **Type inference**
    - **Instruction selection** (converting IR to machine instructions)
    - **Expression simplification** (e.g., `x * 1 → x`)

📌 **Example:**
```
RewritePattern(
  match("arith.muli(%x, 1)"),
  replace("%x")
)
```
This indicates:
- **`x * 1`** is simplified to **`x`**, which is an optimization of the multiplication operation.

---

### **4️⃣ Extensibility and Incremental Lowering**

> "Since we aim for unprecedented extensibility and incremental lowering capabilities, this opens numerous avenues for modeling program transformations as rewrite systems."

**Due to our aim for unprecedented extensibility and incremental lowering, this opens up many possibilities for using rewrite systems to model program transformations.**

- **Extensibility**:
    - Developers can **define new IR dialects and transformation rules** without impacting the core compiler.
    - **For example**: Adding tensor operations could involve writing rules to rewrite tensor operations.
- **Incremental Lowering**:
    - **Rather than lowering high-level IR to machine code all at once, the process is done step by step**:
        - **High-level IR → Low-level IR** (e.g., converting tensor operations to matrix multiplications).
        - **Low-level IR → LLVM IR** (converting matrix operations to LLVM instructions).
        - **LLVM IR → Machine code** (producing the final CPU instructions).

This **layered transformation** makes the compiler more flexible, supporting different hardware architectures.

---

### **5️⃣ Challenges**

> "It also raises interesting questions about how to represent the rewrite rules and strategies, and how to build machine descriptions capable of steering rewriting strategies through multiple levels of abstraction."

**This raises challenges such as how to represent rewrite rules and strategies, and how to build machine descriptions that can guide rewrite strategies across multiple levels of abstraction.**

- **How to represent rewrite rules?**
    - Use **ODS (Operation Definition Specification)** to define rules?
    - Use a **Pattern Match DSL** (similar to MLIR’s PDL)?
- **How to select the appropriate rewrite strategy?**
    - **Which rewrites to apply first?**
    - **Should heuristic search** be employed to select the optimal order of transformations?

These questions are not fully resolved, but MLIR offers **PDL (Pattern Description Language)** to help address these challenges.

---

## **Summary**

1. **Declarative Rewrite Patterns** make IR transformations simpler, analyzable, and extensible.
2. **Rewrite Systems** are widely applied in type inference, instruction selection, etc.
3. **MLIR’s design enables incremental lowering**, allowing more flexible transformations.
4. **Challenges include how to design rewrite rules, select transformation order, and support multi-level IR representations.**


- **Design Principle - Source Location Tracking and Traceability**

This section focuses on **Source Location Tracking and Traceability**, discussing how to retain the **original location information** of operations (such as in the source code) during the compilation process, and how to track all the transformations it undergoes. This addresses the issue of **lack of transparency** in traditional compilers, ensuring that the final generated code can be traced back to the **original source**.

This is particularly important in **safety-critical** and **privacy-sensitive applications**, where compiler optimizations may impact security. The term **WYSINWYX** (What You See Is Not What You eXecute) is introduced, which means that **the security properties of the original code may be lost or altered during optimization**, potentially causing the compiled code to no longer meet expected security standards.

---

## **Glossary and Corresponding Original Text**

---

### **1️⃣ Source Location Tracking**

> The provenance of an operation—including its original location and applied transformations—should be easily traceable within the system.

**Source Location Tracking** refers to preserving the original location information of each IR operation during the compilation process, including details such as:
- **Which source file** the operation originated from
- **Which line and column** it was located on
- **What optimizations** were applied
- **Which passes** have modified it

Why is this important?
- **Improved compilation transparency**: It allows developers to understand the changes made by the compiler.
- **Debugging support**: If the generated code has errors, the origin in the source code can be traced.
- **Security verification**: For **safety-critical applications** (e.g., medical, aviation, finance), the complete **transformation history** is required to ensure no unintended modifications are made.

---

### **2️⃣ Lack-of-Transparency Problem**

> This intends to address the lack-of-transparency problem, common to complex compilation systems, where it is virtually impossible to understand how the final representation was constructed from the original one.

Many modern compilers go through **multiple stages of IR lowering and optimization**, causing the final machine code to differ significantly from the original source code. 
- **This makes it difficult for developers to trace program behavior**, especially in fields that require strict security and certification (e.g., aviation, medical, finance).
- **Example**: 
    - If you write a C program and, after compiler optimizations, some functions are inlined or some variables are eliminated, it becomes hard to relate the final code back to the original, creating a **lack-of-transparency problem**.

---

### **3️⃣ Safety-Critical and Sensitive Applications**

> This is particularly problematic when compiling safety-critical and sensitive applications, where tracing lowering and optimization steps is an essential component of software certification procedures.

- **Safety-Critical Applications**:
    - Aerospace (avionics), autonomous driving, medical devices
    - These applications require **formal verification** to ensure the code's behavior meets expectations.
- **Privacy-Sensitive Applications**:
    - Cryptographic protocols, financial transactions
    - These applications often include **security mechanisms** that could be compromised if the compiler optimizes them away.

📌 **Example**:
- Consider a **cryptographic function designed to prevent timing side-channel attacks**:
    - The compiler might optimize it into a more efficient form, inadvertently leaking timing information to attackers.

```c
if (password == input) {
    perform_computation();
}
```

```c
if (password == input) perform_computation();
```

- With **source location tracking**, it would be possible to check which optimizations affect security properties, preventing such issues.

---

### **4️⃣ WYSINWYX (What You See Is Not What You eXecute)**

> This lack of transparency is known as WYSINWYX in secure compilation.

**WYSIWYG** (What You See Is What You Get) means the displayed content is the final result. **WYSINWYX** (What You See Is Not What You eXecute) reflects how the compiler changes the code so that **the original code does not match the final executed code**.

📌 **Impact of WYSINWYX**:
- **Security defenses may be removed by the compiler**:
    - For instance, some cryptographic algorithms use **random redundant calculations** to defend against attacks, but the compiler might eliminate them as "useless," weakening the security of the system.
- **Timing attacks** may become feasible:
    - Cryptographic code might enforce equal time for operations to prevent side-channel attacks, but after optimization, conditional branches may be introduced, altering execution time and exposing security vulnerabilities.

---

### **5️⃣ Secure and Traceable Compilation**

> One indirect goal of accurately propagating high-level information to the lower levels is to help support secure and traceable compilation.

- **Secure Compilation**:
    - Ensures that optimizations **do not compromise the original code's security**, such as:
        - Not removing security checks
        - Not leaking private information
- **Traceable Compilation**:
    - Enables developers to **fully trace each IR change**, ensuring the compiler's optimizations meet expectations.

---

## **Summary**

1. **Source Location Tracking** allows developers to track all changes made during compilation, improving transparency.
2. **Lack-of-Transparency Problem** makes it difficult to understand how IR is transformed, especially in security-critical applications.
3. **Safety-Critical & Privacy-Sensitive Applications** require a full transformation history to ensure software meets certification standards.
4. **WYSINWYX** issues cause compiled code to potentially lose its original security properties, creating vulnerabilities.
5. **Secure and Traceable Compilation** is a key challenge for future compilers, ensuring both security and traceability of the compilation process.