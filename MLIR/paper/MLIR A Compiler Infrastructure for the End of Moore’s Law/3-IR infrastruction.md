# **🔹Operation Description in MLIR (ODS)**

This section explains **Operation Description (ODS) in MLIR**, focusing on how **TableGen** is used to define operations in MLIR. It introduces core concepts such as **Traits, Arguments, Results, Verifiers, and the C++ code generation mechanism**.

---

## **🔹 Terminology and Corresponding Original Text**

---

### **1️⃣ MLIR Uses TableGen to Describe Operations**

> MLIR uses TableGen-based specification for Operation Descriptions (ODS), defining the structure of an Op and components of its verifier declaratively.  
>
> **TableGen is a data modeling tool intended to help define and maintain records of domain-specific information, used extensively in LLVM.**  

**MLIR uses the `TableGen`-based Operation Description Specification (ODS) to describe operations, enabling a **declarative** definition of an Op’s structure and its verifier logic.**

📌 **What is TableGen?**  

- **TableGen is a data modeling tool in LLVM**, designed to define and manage domain-specific information.  
- **In MLIR, TableGen is used to define operations and rewrite patterns.**  
- **It allows developers to describe operations using a concise DSL (Domain-Specific Language), which is later translated into C++ code.**  

📌 **What is ODS?**  

- **ODS (Operation Description Specification) is a DSL specifically designed for describing operations in MLIR.**  
- **ODS is part of TableGen syntax, but MLIR gives it a specific semantic meaning.**  
- **Ultimately, ODS is converted into C++ code, generating Op classes (including named accessors, verification functions, etc.) that integrate with MLIR.**  

---

### **2️⃣ Core Components of ODS Definitions**  

> Ops are modeled in ODS using the TableGen Op class. Each defined Op has:  
>
> - **A name (unique identifier)**  
> - **A list of traits (describing the properties of the Op)**  
> - **A list of arguments (inputs to the Op, including operands and attributes)**  
> - **A list of results (outputs of the Op)**  
> - **An optional description for improved readability**  
> - **Optional Printer/Parser rules for custom textual representation**  

**In ODS, an operation is a subclass of the `Op` class, consisting of the following components:**  

1. **`name`**: The unique name of the Op, such as `"leaky_relu"`.  
2. **`traits`**: Characteristics of the Op, such as **NoSideEffect** or **SameOperandsAndResultType**.  
3. **`arguments`**: The inputs to the Op, including **operands** and **attributes**.  
4. **`results`**: The outputs of the Op, specifying names and type constraints.  
5. **`description`** (optional): Provides a detailed description of the Op.  
6. **`Printer/Parser`** (optional): Allows the Op to have **a custom textual representation**.  

📌 **Example: ODS Definition of LeakyReLU**  

```mlir
def LeakyReluOp: Op<"leaky_relu",
  [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Leaky ReLU operator";
  let description = [{
    Element-wise Leaky ReLU operator
    x -> x >= 0 ? x : (alpha * x)
  }];
  let arguments = (ins AnyTensor:$input, F32Attr:$alpha);
  let results = (outs AnyTensor:$output);
}
```

Here:  

- **Op Name:** `"leaky_relu"`.  
- **Traits:**  
  - `NoSideEffect` → The Op has no side effects (pure function).  
  - `SameOperandsAndResultType` → The Op’s input and output types must be the same.  
- **Inputs (`arguments`):**  
  - `AnyTensor:$input` → Allows any tensor as input.  
  - `F32Attr:$alpha` → Accepts a `float32` attribute as a parameter.  
- **Outputs (`results`):**  
  - `AnyTensor:$output` → The Op produces a tensor as output.  

---

### **3️⃣ ODS Traits**  

> Op traits can be generic (e.g., “has no side effects”) and dialect- or ODS-specific (e.g., “has a custom exporter”).  
>
> **Traits in ODS may be backed by C++ classes defining the behavior of the trait.**  

**Traits in MLIR ODS define properties of an Op, such as:**  

- **Generic Traits:**  
  - `NoSideEffect` → The Op has no side effects (e.g., `add`).  
  - `SameOperandsAndResultType` → The Op’s input and output types must be the same (e.g., `relu`).  
- **Dialect-Specific Traits:**  
  - Traits like `"has custom exporter"` indicate that the Op requires specialized export logic.  

📌 **Traits Can Correspond to C++ Classes**  

- **Traits are not limited to ODS; they can be implemented in C++.**  
- **For example, `NoSideEffect` might correspond to a C++ class `NoSideEffectTrait`, which defines whether an Op affects the global state.**  

---

### **4️⃣ ODS Type Constraints**  

> Type constraints check properties of the type of arguments/results and are user/dialect extensible.  
>
> **MLIR infrastructure also provides numerous pre-defined type constraints, such as:**  
>  
> - **"any type"**  
> - **"tensor with an element satisfying the given constraint"**  
> - **"vector of given rank"**  

**ODS allows type constraints on inputs (`arguments`) and outputs (`results`) to ensure that Ops have valid types in MLIR IR.**  

MLIR provides several predefined type constraints, such as:  

- **`AnyType`**: Accepts any type.  
- **`TensorOf<ElementType>`**: Requires a tensor with a specific element type.  
- **`VectorOf<Rank>`**: Requires a vector of a specified rank.  

📌 **Example**  

```mlir
let arguments = (ins TensorOf<F32>:$input);
let results = (outs TensorOf<F32>:$output);
```

This means:  

- The Op **only accepts `tensor<f32>` as input** and **must produce `tensor<f32>` as output**.  

---

### **5️⃣ ODS Can Automatically Deduce the Output Type of an Op**  

> ODS also has limited support for automatically deducing the return type of results from operands using constraints induced by traits.  

ODS **can automatically infer an Op’s output type based on the constraints defined by its Traits.** For example:  

- If a trait specifies that "the Op’s input and output types are the same" (`SameOperandsAndResultType`), the output type can be derived from the input.  
- If `relu` takes `tensor<f32>` as input, the output must also be `tensor<f32>`.  

📌 **Example**  

```mlir
def LeakyReluOp: Op<"leaky_relu",
  [SameOperandsAndResultType]> {
  let arguments = (ins TensorOf<F32>:$input, F32Attr:$alpha);
  let results = (outs TensorOf<F32>:$output);
}
```

Here, **the `output` type does not need to be explicitly specified** because `SameOperandsAndResultType` ensures that it is automatically inferred as `tensor<f32>`.  

---

## **🔹 Summary**  

1. **MLIR uses TableGen (ODS) to define operations, making Op definitions concise and automatically generating C++ code.**  
2. **An Op in ODS consists of `name`, `traits`, `arguments`, and `results`, with optional `description` and `printer/parser`.**  
3. **Traits define the properties of an Op and can be generic (e.g., `NoSideEffect`) or dialect-specific. Traits can also be backed by C++ classes.**  
4. **ODS allows type constraints to ensure that an Op's arguments and results have valid types.**  
5. **ODS can automatically infer an Op’s output type based on defined traits, improving IR consistency and safety.**  


---

# **🔹 2 Declarative Rewrites (DRR)**

This section explains **how MLIR uses Declarative Rewrite Rules (DRR) for IR transformations**, where **DRR is a TableGen-based DSL** used to describe **pattern matching and transformation rules for Directed Acyclic Graphs (DAGs)**.

---

## **📌 What is DRR?**

**Most MLIR optimizations and transformations involve modifying operations**, and **some transformations can be complex, while others can be implemented simply through "pattern matching + replacement"**—this is the **core idea behind DRR**.

MLIR provides a **graph rewriting framework** and uses the **DRR (Declarative Rewrite Rule) system** to allow developers to **define equivalence rules between operations in a declarative manner**.

### **Characteristics of DRR:**
1. **Similar to ODS, DRR is also part of the TableGen DSL**, used to describe transformations between a "Source DAG" and a "Target DAG."
2. **DRR allows defining DAG patterns**, which include:
    - **Constraints**, ensuring the validity of operation matching.
    - **Priority (Benefits)**, determining the order in which patterns are applied.
3. **DRR enables capturing operation inputs (arguments) and reusing them in transformations.**
4. **Ultimately, DRR is converted into C++ code**, integrating with manually written C++ pattern rewriters.

---

## **📌 Basic Structure of DRR**

> DRR expresses source and target DAG patterns along with constraints and benefits for pattern prioritization.  
>  
> **Conceptually, DRR expresses equivalence of DAGs under specific constraints.**

### **🔸 Two Main Parts of DRR**
1. **Source DAG**
    - Defines the **matching pattern**, specifying the IR fragment to be replaced.
    - The nodes correspond to **operations defined in ODS** and may contain **arguments and type constraints**.
2. **Target DAG**
    - Defines **how to replace the Source DAG with a new IR fragment**.
    - The target DAG can reference **matched values from the source DAG** or **construct new operations** to form equivalent computations.

---

## **📌 Example of DRR**

> **Figure 6**: Declarative graph rewrite rule transforming a `LeakyReluOp` into a `CmpFOp` followed by a `SelectOp`.

This example demonstrates **how `LeakyReluOp` is transformed into an equivalent IR representation using `CmpFOp` + `SelectOp`.**

### **🔸 `LeakyReluOp` (Source Operation)**

- `LeakyReluOp` is an element-wise operation that:
    - **Inputs:** `$arg` (a tensor) and `$alpha` (a float constant).
    - **Computation rule:**
        - **If `arg >= 0`, the output is `arg`.**
        - **Otherwise, the output is `alpha * arg`.**

### **🔸 Transformed IR (Target Operations)**

Here, `LeakyReluOp` is transformed into:

1. **`CmpFOp`** → Compares `arg` with `0.0` to check if it is greater than or equal.
2. **`SelectOp`** → Selects the output based on the `CmpFOp` result:
    - If `arg >= 0`, the output is `arg`.
    - Otherwise, the output is `alpha * arg` (produced using `ConstantOp` for `alpha`).

---

### **🔹 DRR Definition in Code**

```tablegen
def : Pattern<
  // Source DAG (matching LeakyReluOp)
  (LeakyReluOp $arg, F32Attr:$alpha),

  // Target DAG (replace with CmpFOp + SelectOp)
  [(SelectOp
    (CmpFOp CMPF_P_ODT
      $arg
      (ConstantOp ConstantAttr<F32Attr, "0.">))
    $arg,
    (ConstantOp $alpha)
  )]
>;
```

### **🔸 Breakdown of the DRR Code**
1. **Matching `LeakyReluOp`**
    
    ```tablegen
    (LeakyReluOp $arg, F32Attr:$alpha)
    ```
    
    - **Matches `LeakyReluOp` and captures `$arg` and `$alpha`.**
    - `$arg` represents the input tensor, and `$alpha` is a `float32` attribute.
    
2. **Replacing it with `CmpFOp` + `SelectOp`**
    
    ```tablegen
    [(SelectOp
       (CmpFOp CMPF_P_ODT
         $arg
         (ConstantOp ConstantAttr<F32Attr, "0.">))
       $arg,
       (ConstantOp $alpha)
    )]
    ```
    
    - **Step 1**: `CmpFOp` **compares `$arg` with `0.0`**.
    - **Step 2**: `SelectOp` **chooses the output based on the comparison**:
        - **If `arg >= 0`, it returns `$arg`.**
        - **Otherwise, it returns `$alpha * arg`** (via `ConstantOp` for `$alpha`).

---

## **📌 Features of DRR**

1. **High Readability & Easy Understanding**
    - **Uses a declarative (DSL-based) approach to describe IR transformations**, avoiding manual C++ coding.
    - **Clearly separates matching patterns from replacement patterns**, improving development efficiency.
2. **High Extensibility**
    - **Supports dynamic constraints** to match more complex patterns.
    - **Can be combined with manually written C++ pattern rewriters** for special cases.
3. **Optimization Selection Mechanism**
    - **Patterns can have priorities (`benefits`)**, ensuring **better transformation rules are applied first**.
4. **Eventually Transformed into C++ Code**
    - **DRR patterns are converted into C++ code** and used in MLIR’s pattern rewrite passes.

---

## **🔹 Summary**

1. **DRR (Declarative Rewrite Rule) is a declarative system in MLIR for defining operation transformations** using **DAG pattern matching + replacement**.
2. **DRR consists of two parts:**
    - **Source DAG (Matching Pattern)** → Specifies the IR fragment to be replaced.
    - **Target DAG (Replacement Pattern)** → Specifies the new IR fragment, which may contain new operations.
3. **Advantages of DRR:**
    - **Improves readability, reducing the need for manual C++ coding.**
    - **Captures operation parameters and reuses them in transformations.**
    - **Can be combined with handwritten C++ logic for greater flexibility.**
4. **DRR is eventually converted into C++ code** and used in MLIR’s pattern rewrite passes.

---

## **🔹 3 Pass Manager**

This section explains **the design and features of the MLIR Pass Manager**, focusing on how it supports **passes at different levels** and enables **parallel compilation**.

---

## **📌 What is a Pass Manager?**

In a **compiler architecture, a pass is responsible for transforming or analyzing the IR**. The **Pass Manager organizes and executes these passes**, ensuring that **optimizations and transformations occur in the correct order**.

Generally:

- **Passes operate at different levels**:
  - **Module-Level Pass**: Analyzes and transforms the entire module.
  - **Function-Level Pass**: Optimizes individual functions.
  - **Loop-Level Pass**: Focuses on loop optimizations (e.g., unrolling, vectorization).
- In traditional compilers (such as LLVM), **the Pass Manager is designed to handle these specific levels**.

However, **MLIR adopts a more generic design**:

- **Module and Function are not special concepts but just general Ops**.
- **The MLIR Pass Manager can run on any Op, without being restricted to predefined levels**.

---

## **📌 Features of the MLIR Pass Manager**

### **🔸 1. Passes at Arbitrary Op Levels**

> MLIR does not have a fixed set of pass granularities. Instead, it operates on arbitrary Ops at arbitrary levels of nesting.

- **Traditional compilers (like LLVM) can only manage passes at predefined levels (Module, Function, Loop)**.
- **The MLIR Pass Manager can execute passes on any Op, including those inside nested regions**.
- **Supports nested structures**:
  - For example, **a pass can be applied at the `ModuleOp` level, but also directly inside `std.func` on `AffineForOp`**.

💡 **This flexible design makes MLIR more adaptable to various compilation scenarios, such as DSL transformations and machine learning graph optimizations.**

---

### **🔸 2. Support for Parallel Compilation**

> MLIR supports concurrent traversal and modification of the IR for faster compilation.

- **Modern CPUs feature multi-core architectures, so the Pass Manager must support parallel execution to speed up compilation**.
- MLIR’s **Pass Manager allows different parts of the IR to be processed in parallel**, as long as they **are independent and do not interfere with each other**.

---

### **🔸 3. Ensuring Safe Parallel Compilation**

> The "isolated-from-above" property ensures safe parallel execution.

**MLIR ensures safe parallel execution of passes through the "Isolated-from-above" property**:

- This means that **certain Ops (such as `std.func`) contain IR that does not interfere with the outer IR through SSA use-def chains**.
- In other words:
  - **SSA use-def chains (variable definition-use relationships) cannot cross region boundaries**.
  - **This allows each function (`std.func`) to be processed independently, without affecting the optimization and transformation of other functions**.

💡 **This mechanism ensures that the MLIR Pass Manager can safely process IR in parallel, avoiding race conditions.**

---

### **🔸 4. Why Doesn’t MLIR Have Whole-Module Use-Def Chains?**

> Unlike LLVM, MLIR does not feature whole-module use-def chains. Instead, global objects are referenced through symbol tables.

- In **LLVM, SSA use-def chains can span the entire module**, which makes certain optimizations (such as whole-module analysis) difficult because:
  - **Tracking all use-def relationships requires scanning the entire module, affecting performance**.
  - **Parallel optimizations become harder since use-def chains can span across functions or even across modules**.
- **MLIR, on the other hand, avoids whole-module use-def chains** by:
  - **Using a symbol table to reference global objects**.
  - **Implementing constants as Ops with attributes instead of simple variable values**.

💡 **This design allows the MLIR Pass Manager to manage IR more efficiently and support parallel compilation.**

---

## **📌 Example: Using the MLIR Pass Manager**

### **🔸 Defining a Pass Pipeline**

In MLIR, a pass pipeline can be managed through the `mlir-opt` command or via the C++ API. For example:

```
mlir-opt my_input.mlir --pass-pipeline="builtin.module(func.func(my-pass))"
```

This means:

1. **Start executing passes from `builtin.module`**.
2. **Enter `func.func` (inside the function) and apply `my-pass`**.
3. **This allows transformations to be applied only inside functions without affecting the entire module**.

---

### **🔸 Pass Pipeline in C++**

The MLIR Pass Manager can also be used in C++:

```cpp
PassManager pm(&context);
pm.addPass(createCanonicalizerPass());  // Add Canonicalization Pass
pm.addPass(createInlinerPass());        // Add Function Inlining Pass
pm.addPass(createSymbolDCEPass());      // Add Symbol DCE (removes unused symbols)

// Execute the Pass Pipeline
if (failed(pm.run(moduleOp))) {
  llvm::errs() << "Pass execution failed!\n";
  return 1;
}
```

Here:

- **`createCanonicalizerPass()`**: Performs **canonicalization**, simplifying the IR.
- **`createInlinerPass()`**: Performs **function inlining**.
- **`createSymbolDCEPass()`**: Eliminates unused **symbols**.

---

## **🔹 Summary**

1. **The MLIR Pass Manager organizes and executes IR passes**. Unlike LLVM, **it does not restrict passes to specific levels (Module/Function/Loop) but instead allows passes to run on arbitrary Ops**.
2. **Supports parallel compilation**:
   - **Uses the "Isolated-from-above" property** to ensure that functions (`std.func`) can be processed independently, without affecting external IR.
   - Allows **passes to run in parallel**, improving compilation efficiency.
3. **Avoids whole-module use-def chains to improve scalability**:
   - **Global objects are managed through a symbol table** instead of relying on a use-def chain.
   - **Constants are implemented using attributes**, preventing cross-function use-def dependencies.

This makes **the MLIR Pass Manager highly flexible** and enables **efficient execution of pass pipelines in multi-core environments** 🚀.

## **🔹 4 Round-Trippable Textual IR Form**  

MLIR's IR has a **textual representation** that **directly corresponds to its in-memory structure**, providing **good readability, debugging capabilities, and ease of testing**.  

---

## **📌 Why Do We Need a Textual IR Form?**  

In **compiler development and optimization, IR readability and visibility are crucial**. The benefits of a textual IR form include:  

1. **Easier Development and Debugging**  
   - Allows developers to **directly inspect IR transformations**, avoiding black-box behavior.  
   - **Readable IR makes debugging more intuitive**, e.g., `mlir-opt` allows developers to examine the output of each pass.  
2. **Improved Testing**  
   - **Each pass’s input and output are textual IR**, ensuring **consistent behavior between passes**.  
   - Enables testing individual passes without running the entire pipeline.  
3. **Support for Manual IR Writing**  
   - This makes IR testing easier—developers can manually write test cases without relying on the frontend to generate IR.  

---

## **📌 How Does MLIR Improve IR Readability?**  

**Traditional IR formats (Raw IR)**, like those in **Figure 4**, tend to be verbose and harder to read. For example:  

```mlir
%1 = "toy.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
```  

While this format explicitly describes the IR operation, **it has lower human readability**.  

### **🔸 Custom IR Formatting (Custom Printing & Parsing)**  

MLIR **allows dialects to define custom IR printing and parsing formats**, making IR more concise:  

```mlir
%1 = toy.add %arg0, %arg1 : tensor<4xf32>
```  

This improves readability while preserving full syntactic information.  

---

## **📌 Round-Trippable IR**  

> Both textual and in-memory representations are fully round-trippable.  

### **🔸 What Is Round-Trippability?**  

- **The textual IR and in-memory IR can be converted back and forth without losing any information.**  
- This means:  
  1. **IR can be serialized into text** (print to text).  
  2. **Textual IR can be parsed back into in-memory IR** (parse back to in-memory IR).  
  3. **The parsed IR is functionally equivalent to the original in-memory IR.**  

### **🔸 Why Is This Important?**  

1. **Ensures Passes Behave Consistently When Run Independently**  
   - Since **IR has no hidden state**, running a pass separately should produce the same result as running it within a full pass pipeline.  
2. **Enhances IR Testing Reliability**  
   - Passes can be tested individually, for example:  

     ```sh
     mlir-opt input.mlir --canonicalize -o output.mlir
     ```

     This ensures that the `canonicalize` pass works correctly without being affected by other passes.  
3. **Enables Manual IR Writing for Pass Testing**  
   - Developers can directly write:  

     ```mlir
     func @test(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
       %1 = toy.add %arg0, %arg1 : tensor<4xf32>
       return %1 : tensor<4xf32>
     }
     ```

     This allows pass testing **without needing a frontend to generate IR**.  

---

## **📌 Example of Round-Trippable IR**  

Consider the following IR:  

```mlir
func @relu(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %1 = "toy.leaky_relu"(%arg0) {alpha = 0.01} : (tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
```  

We can run:  

```sh
mlir-opt relu.mlir --convert-to-affine-loops -o optimized.mlir
```  

**The result remains valid textual IR and can be re-parsed into in-memory IR, ensuring consistency across passes.**  

---

## **🔹 Summary**  

1. **MLIR provides a textual IR representation, making IR readable, testable, and debuggable.**  
2. **MLIR allows custom IR formatting, improving readability and avoiding excessive verbosity.**  
3. **Textual IR and in-memory IR are fully round-trippable, ensuring consistency between pass testing and pipeline execution.**  
4. **Developers can manually write IR for testing passes, without depending on frontend-generated IR.**



---

# **📌 5 Documentation**  

During the development of MLIR **dialects, operations (Ops), and interfaces**, MLIR **automatically generates documentation from ODS (Operation Description Specification)**. This not only reduces the burden of manual documentation but also ensures that **the documentation remains consistent with the actual behavior**.  

### **🔸 What Does the Documentation Include?**  

1. **Summary** – A **one-line description** of each Op, providing a quick understanding of its purpose.  
2. **Detailed Description** – A more **comprehensive explanation** to help developers understand the Op's behavior.  
3. **Arguments & Results** –  
   - **Records the input (operands) and output (results) type constraints** for each Op.  
   - Example:  

   The following ODS description not only defines the `LeakyReluOp` but also automatically generates its documentation, including:  

   ```mlir
   def LeakyReluOp: Op<"leaky_relu", [NoSideEffect, SameOperandsAndResultType]> {
     let summary = "Leaky ReLU operator";
     let description = [{ Element-wise Leaky ReLU operator x -> x >= 0 ? x : (alpha * x) }];
     let arguments = (ins AnyTensor:$input, F32Attr:$alpha);
     let results = (outs AnyTensor:$output);
   }
   ```

   - `input` must be **any tensor (AnyTensor)**.  
   - `alpha` must be **a single-precision floating-point attribute (F32Attr)**.  
   - The output **must have the same type as the input**.  

### **🔸 How Does Documentation Stay in Sync with Execution Behavior?**  

- **ODS serves as both the source of verification rules and the source of documentation**, meaning:  
  - **Validation logic and documentation are updated together**, reducing the likelihood of errors.  
  - **No need to manually write API documentation—developers can rely on ODS-generated documentation**.  

---

# **📌 6 Verifiers**  

MLIR uses **verifiers to ensure the correctness of IR**, which is crucial for **ensuring correctness during IR transformations and optimizations**.  

### **🔸 Goals of Verification**  

1. **Ensure the structural correctness of IR**  
   - Types must match exactly.  
   - **Each value must be defined only once** (SSA compliance).  
   - **Use-def chains must follow SSA rules**.  
   - **Blocks must end with a terminator operation**.  
   - **Names in symbol tables must be unique**.  
2. **Ensure the correctness of specific Ops**  
   - **Each Op can define its own structural and semantic checks**.  
   - Examples:  
     - **A binary operation must have exactly two operands**.  
     - **Certain Ops only accept specific input types**.  
     - **Some Ops require specific attributes or regions to function**.  
3. **Enforce dialect-specific constraints**  
   - Some dialects restrict **which attributes can be applied to certain Ops**.  
   - Example:  
     - **An Op within a dialect can only use types defined by that dialect** (e.g., `tensor<4xf32>` can only be used within the `toy` dialect).  

### **🔸 What Happens When Verification Fails?**  

If an error occurs during verification, it is treated as an **invariant violation**, causing **immediate compilation termination**.  

---

## **📌 How the Verifier Works**  

When running an MLIR pass, the verifier performs:  

1. **Basic IR structure checks (Structural Check)**.  
2. **Op-specific verification (Op-Specific Verification)**.  
3. **Dialect-specific constraint checks (Dialect-Specific Constraints)**.  

For example, given the following incorrect IR:  

```mlir
%1 = "toy.add"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
```  

The `"toy.add"` Op is missing an operand, so the verifier will report an error:  

```
error: 'toy.add' op expected 2 operands, but got 1
```  

This will **immediately abort compilation**.  

---

# **📌 Summary**  

✅ **MLIR can automatically generate documentation from ODS, ensuring developers get accurate API information.**  
✅ **Verifiers ensure the structural correctness of IR, preventing passes from generating invalid IR.**  
✅ **Verifiers check basic IR rules, Op-specific rules, and dialect constraints, aborting compilation upon detecting errors.**  

