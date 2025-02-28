### **IR Design Concepts**

| MLIR Concept | Analogy |
| --- | --- |
| **Operations (Op)** | Python’s "everything is an object" |
| **SSA (Static Single Assignment)** | Rust’s "immutable variables" |
| **Attributes** | Function's "default arguments" |
| **Regions & Blocks** | C/Python's `{}` blocks |
| **Symbols & Symbol Table** | C++’s "namespace" |
| **Dialects** | Different "libraries" |
| **Type System** | C/Rust/Python's type system |


This section mainly describes **MLIR (Multi-Level Intermediate Representation)**'s **intermediate language design**, but it contains many technical details. To explain it in a simpler way, using more familiar concepts, we can understand it like this:

---

### **1. Operations (Ops) Are the Basic Unit of MLIR**

In MLIR, **everything is an "Op"**. Whether it's a basic mathematical operation, loop, function, or even a module, everything is considered an "Operation" (Op).

👉 Analogy: Just like in Python, where everything is an object, in MLIR, everything is treated as an "Op."

Example:

```
%result = "std.addf"(%a, %b) : (f32, f32) -> f32
```

Here, `std.addf` represents an **addition operation (Add Op)**, taking two floating-point numbers (f32) as input and outputting a floating-point number (f32).

---

### **2. MLIR Uses SSA Form**

MLIR uses **Static Single Assignment (SSA)**, which means:

- **Each variable can only be assigned once** (it cannot be reassigned).
- **The program’s control flow is clear and traceable**.

👉 Analogy: This is like Rust’s **immutable variables**, where once declared, the value cannot be changed.

Example:

```
%0 = "affine.load"(%A, %i) : (memref<?xf32>, index) -> f32
%1 = "affine.load"(%B, %j) : (memref<?xf32>, index) -> f32
%2 = "std.mulf"(%0, %1) : (f32, f32) -> f32
```

Here, `%0`, `%1`, and `%2` can only be assigned once and cannot be reassigned.

---

### **3. Attributes Are Compile-Time Information**

Attributes are **compile-time fixed data** used to describe the behavior of an Op, such as:

- Constant values
- Loop bounds
- Transformation rules (e.g., Affine Maps)

👉 Analogy: Like **default arguments** in functions, which are decided during compilation.

Example:

```
{lower_bound = () -> (0), step = 1 : index, upper_bound = #map3}
```

This represents a loop with **starting point 0, step size 1, and upper bound as map3** (a pre-defined rule).

---

### **4. Regions and Blocks Represent Control Flow**

MLIR manages **control flow** through **Regions** and **Blocks**.

- A **Region** is a block collection with its own variable scope.
- A **Block** is a "basic block," containing multiple Ops, ending with a **terminator**.

👉 Analogy:

- **Region** is like the **scope inside a function**.
- **Block** is like the `{}` or `:` indentation block in C/Python.

Example:

```
"affine.for"(%arg0) ({
  ^bb0(%arg4: index):
    %0 = "affine.load"(%arg1, %arg4) : (memref<?xf32>, index) -> f32
    %1 = "affine.load"(%arg2, %arg4) : (memref<?xf32>, index) -> f32
    %2 = "std.mulf"(%0, %1) : (f32, f32) -> f32
    "affine.store"(%2, %arg3, %arg4) : (f32, memref<?xf32>, index) -> ()
  "affine.terminator"() : () -> ()
})
```

This is an **Affine For loop**:

1. The beginning `affine.for` represents a loop within a fixed range.
2. The block `^bb0(%arg4: index)` represents the content executed in each iteration.
3. The `affine.terminator` indicates the end point of the loop.

---

### **5. Symbols and Symbol Table**

MLIR supports a **Symbol mechanism**, allowing globally named objects such as variables and functions.

These symbols are stored in the **Symbol Table** and can be used across different regions.

👉 Analogy:

- **Symbols** are like **variable names**, which can be referenced in different places.
- **Symbol Table** is like a **C++ namespace**, organizing different functions or variables.

Example:

```
module {
  func @my_function(%arg0: i32) -> i32 {
    %result = "std.addi"(%arg0, %c1) : (i32, i32) -> i32
    return %result : i32
  }
}
```

Here, `@my_function` is a symbol (symbol), which can be called elsewhere.

---

### **6. Dialects**

MLIR doesn’t have a fixed syntax but instead uses **Dialects** to extend itself.

Different Dialects can define their own Ops, types, and rules, for example:

- **affine**: Supports mathematical mappings (Affine Maps), optimizes loops
- **std**: Standard operations (addition, subtraction, multiplication, etc.)
- **LLVM**: Corresponds to LLVM IR, facilitating translation to machine code

👉 Analogy:

- **Dialect** is like a **different programming library**, where each library has its own API and methods.

Example:

```
%result = "std.mulf"(%a, %b) : (f32, f32) -> f32
```

Here, `"std.mulf"` comes from the `std` dialect, representing floating-point multiplication.

---

### **7. Type System**

MLIR variables have explicit types, such as:

- `i32`: 32-bit integer
- `f32`: 32-bit floating-point number
- `memref<?xf32>`: Memory reference (Memory Reference)

👉 Analogy:

- **MLIR types** are similar to the **type systems** in C/Rust/Python, where you can explicitly define variable types.

---

## **1. Operation, Attributes, Location Information**

This section mainly introduces **MLIR (Multi-Level Intermediate Representation) Operations (Ops)**, which are the **basic semantic units of MLIR**. In MLIR, almost everything—from **instructions**, **functions**, to **modules**—is represented as an **Op (Operation)**.

This design allows users to **extend MLIR** by adding their own custom-defined Ops, without being restricted to a fixed instruction set. **Compiler passes** conservatively handle unknown Ops and use **Traits**, **privileged operation hooks**, and **optimization interfaces** to understand the semantics of these Ops.

---

## **🔹 Terminology Explanation and Corresponding Original Text**

---

### **1️⃣ Operation (Op)**

> The unit of semantics in MLIR is an “operation”, referred to as Op. Everything from “instruction” to “function” to “module” are modeled as Ops in this system.

**The semantic unit in MLIR is the "operation (Op)," and everything—from "instruction," to "function," to "module"—is modeled as an Op in this system.**

- **Op (Operation)** is the **core unit** in MLIR, responsible for expressing program semantics.
- **MLIR is not based on a fixed instruction set but allows users to extend their own Operations**, making it more flexible than LLVM IR.

📌 **Example:**
In LLVM IR, we have fixed instructions like:

```llvm
%1 = add i32 %a, %b
```

But in MLIR, **addition (Addition) can be a user-defined Op, such as:**

```
%1 = "my_dialect.add"(%a, %b) : (i32, i32) -> i32
```

Here, `"my_dialect.add"` is an **Op** belonging to the `"my_dialect"` dialect, which can be user-defined.

---

### **2️⃣ MLIR’s Op Flexibility**

> MLIR does not have a fixed set of Ops, but allows (and encourages) user-defined extensions—compiler passes treat unknown Ops conservatively.

**MLIR does not have a fixed set of Ops, but allows (and encourages) users to define extensions—compiler passes treat unknown Ops conservatively.**

- **LLVM IR has a fixed set of instructions** (e.g., `add`, `mul`, `load`, `store`).
- **MLIR, on the other hand, is open-ended and allows users to add their own Ops**.
- **This design makes MLIR adaptable to various domains**, such as AI, machine learning, graphics, and mathematical computations.

📌 **Example**

- **AI/Machine Learning Dialect (ML Dialect):**

Here, `"tensor.matmul"` is a matrix multiplication operation that is not in the traditional LLVM IR instruction set.

```
%1 = "tensor.matmul"(%a, %b) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
```

---

### **3️⃣ Operation Structure**

> Ops have a unique opcode, which, textually, is a string identifying its dialect and the operation. Ops take and produce zero or more values, called operands and results respectively, and these are maintained in SSA form.

**Ops have a unique opcode, which, textually, is a string identifying its dialect and operation type. Ops take and produce zero or more values, called operands and results respectively, and these are maintained in SSA form.**

- **Opcode**: Identifies the operation, such as `"tensor.matmul"`, `"arith.addf"`.
- **Operands**: Input values to the Op.
- **Results**: Output values from the Op.
- **Op is maintained in Static Single Assignment (SSA) form**, similar to LLVM IR.

📌 **Example**

```
%results:2 = "d.operation"(%arg0, %arg1) : (!d.type, !d.type) -> (!d.type, !d.other_type)
```

This indicates `"d.operation"`:

- **Has 2 operands** (`%arg0`, `%arg1`).
- **Produces 2 results** (`%results:2`).
- **Uses MLIR's SSA form to manage values**.

---

### **4️⃣ Attributes**

> An MLIR attribute is structured compile-time static information, e.g., integer constant values, string data, or a list of constant floating point values.

**MLIR attributes are structured compile-time static information, such as integer constants, string data, or lists of constant floating point values.**

- **Attributes are additional static information for Ops**. They don't affect the execution of the Op but are useful for optimization.
- **Similar to LLVM IR’s metadata, but more flexible**.
- **Attributes are key-value pairs** that can be defined within Ops.

📌 **Example**

```
"affine.for" () {
  lower_bound = () -> (0),
  step = 1 : index,
  upper_bound = #map3
}
```

Here, `"affine.for"` is a loop that uses **attributes** to define:

- **lower_bound**: `(0)`, indicating the starting index is `0`.
- **step**: `1`, indicating the increment is `1` per iteration.
- **upper_bound**: `#map3`, indicating the iteration stop condition.

---

### **5️⃣ Location Information**

> MLIR provides a compact representation for location information, and encourages the processing and propagation of this information throughout the system.

**MLIR provides a compact representation to store location information and encourages processing and propagating this information throughout the system.**

- **Location Information helps the compiler track where an Op originates from**.
- **This is useful for debugging, error diagnosis, debugging tools, and optimization**.
- **MLIR provides various formats for location information, such as LLVM-style file-line-column addresses, DWARF debug info**.

📌 **Example**

```
%1 = "some.op"(%0) { loc = FileLineColLoc("source.mlir", 12, 34) } : (!type) -> (!type)
```

This means:

- **This Op originates from the file "source.mlir" at line 12, column 34**.
- **Helps with error diagnosis and backtracking**.

---

### **🔹 Summary**

1. **The basic unit of MLIR is Operation (Op), and everything (instructions, functions, modules) is an Op.**
2. **MLIR doesn’t have a fixed set of instructions and allows users to extend their own Ops, making it suitable for different domains (AI, mathematical operations, etc.).**
3. **Ops consist of "opcode, operands, results, and attributes" and maintain SSA form.**
4. **Attributes are static information within Ops, used to store constants, index ranges, etc.**
5. **Location Information helps the compiler track the source of an Op and is crucial for debugging and optimization.**


## **2.Region and Blocks**

This section introduces **Regions** and **Blocks** in **MLIR (Multi-Level Intermediate Representation)**, which are the core mechanisms for supporting **nested structures** and **control flow** in MLIR.

In MLIR, **an Operation (Op) can contain multiple Regions, and each Region consists of multiple Blocks.** Blocks contain multiple Ops, and these Blocks can form a **Control Flow Graph (CFG).**

---

### **🔹 Terminology and Corresponding Original Text**

---

### **1️⃣ Region**

> An instance of an Op may have a list of attached regions. A region provides the mechanism for nested structure in MLIR: a region contains a list of blocks, and a block contains a list of operations (which may contain regions).

**An Op can have one or more attached Regions. A Region provides the nested structure mechanism in MLIR: a Region contains multiple Blocks, and a Block contains multiple Ops (which may contain Regions).**

📌 **Key Points**
- **Regions provide the nested mechanism** in MLIR, allowing Ops to have internal Blocks.
- **A Region contains Blocks, and Blocks contain Ops**, making MLIR's structure more flexible than LLVM IR.
- **The semantics of a Region are defined by the Op it belongs to**, meaning **the Op determines how the Region behaves**.

📌 **Example**

```
%result = "d.operation"() ({
  ^bb0:
    %v1 = "some.op"() : () -> i32
    "some.terminator"() : () -> ()
}) : () -> i32
```

Here, `"d.operation"` is an **Op** that contains a **Region**:
- **The Region contains one Block (`^bb0`)**.
- **The Block contains two Ops (`some.op` and `some.terminator`)**.
- This allows MLIR to support higher-level structures (like functions, control flow, etc.) within Ops.

---

### **2️⃣ Block**

> As with attributes, the semantics of a region are defined by the operation they are attached to, however the blocks inside the region (if more than one) form a Control Flow Graph (CFG).

**The semantics of a Region are defined by the Op it belongs to, but if there are multiple Blocks within the Region, these Blocks form a "Control Flow Graph (CFG)."**

📌 **Key Points**
- **Blocks are the fundamental unit within a Region, and each Block contains multiple Ops.**
- **If there are multiple Blocks in a Region, they form a "Control Flow Graph (CFG)."**
- **The control flow between Blocks is determined by Terminator Ops** (such as `switch`, `br`, etc.).

📌 **Example**

```
"affine.for" () ({
  ^entry(%arg4 : index):
    "some.op"() : () -> ()
    "affine.yield"() : () -> ()
}) : () -> ()
```

Here, `"affine.for"` is a typical example of a **Region with Blocks**:
- **The Region contains one Block (`^entry`)**.
- **The Block contains one `some.op` and ends with `affine.yield`**.
- **`%arg4` is the argument (SSA variable) for this Block**.

---

### **3️⃣ Terminator Operation**

> Each block ends with a terminator operation, that may have successor blocks to which the control flow may be transferred.

**Each Block ends with a "Terminator Operation" that may transfer the control flow to other Blocks.**

📌 **Key Points**
- **A Terminator Op is the last Operation in a Block**, determining which Block comes next.
- Common Terminator Ops include:
    - `"branch"` (`br`) → **Unconditional jump**
    - `"cond_br"` → **Conditional jump**
    - `"switch"` → **Multi-way selection**
    - `"return"` → **Return**
- **These Terminator Ops define how the Blocks form the control flow graph (CFG).**

📌 **Example**

```
^bb0:
  %cond = "some.compare"() : () -> i1
  "cond_br"(%cond, ^bb1, ^bb2) : (i1) -> ()
^bb1:
  "some.op"() : () -> ()
  "br"(^bb2) : () -> ()
^bb2:
  "return"() : () -> ()
```

Here:
- **`^bb0` contains `cond_br`, which decides whether to jump to `^bb1` or `^bb2` based on `%cond`**.
- **`^bb1` has `br`, which unconditionally jumps to `^bb2`**.
- **`^bb2` has `return`, indicating the end of the function**.

This forms a **Control Flow Graph (CFG)**!

---

### **4️⃣ Block Arguments**

> Each block has a (potentially empty) list of typed block arguments, which are regular values and obey SSA.

**Each Block can have (or not) a list of typed "Block Arguments," which are regular SSA variables.**

📌 **Key Points**
- **Block Arguments are defined at the start of a Block, similar to function parameters.**
- **These arguments are passed through Terminator Ops.**
- **This design avoids the use of `phi` nodes in LLVM IR**, making the control flow clearer.

📌 **Example**

```
^entry(%arg4: index):
  "some.op"(%arg4) : (index) -> ()
```

Here:
- **The `^entry` Block has an argument `%arg4`**.
- **`some.op` directly uses this argument**.

This is more intuitive than LLVM IR's `phi` nodes.

---

### **🔹 MLIR vs. LLVM IR Advantages**

| Feature | MLIR | LLVM IR |
| --- | --- | --- |
| **Nested Structure** | ✅ **Region (supports nesting)** | ❌ (Flat CFG) |
| **Block Arguments** | ✅ **Block Arguments** | ❌ (Uses `phi` nodes) |
| **Terminator Ops** | ✅ `switch`, `cond_br`, `br` | ✅ (Similar to `br`, `switch`) |
| **Extensibility** | ✅ (Supports multiple dialects and Ops) | ❌ (Fixed instruction set) |

MLIR's **Regions and Blocks** design is **more flexible than LLVM IR**, particularly suited for:
1. **Machine Learning (ML) Model Optimization**
2. **Custom Compilers and DSLs**
3. **High-Level Language Translations**

---

### **🔹 Summary**

1. **Ops can contain Regions, Regions contain Blocks, and Blocks contain Ops.**
2. **Control flow between Blocks is determined by Terminator Ops (e.g., `cond_br`, `br`), forming a Control Flow Graph (CFG).**
3. **MLIR uses Block Arguments to manage variables, replacing LLVM IR's `phi` nodes, making control flow more intuitive.**
4. **Regions allow MLIR to support nested structures, making it more suitable for high-level language and DSL translations.**


## **3.Value Dominance and Visibility**

This section discusses **Value Dominance** and **Visibility** in **MLIR (Multi-Level Intermediate Representation)**. These concepts relate to how **MLIR ensures the rules of SSA (Single Static Assignment)** and how the scope of variables in different regions is controlled.

---

## **🔹 Definitions and Corresponding Text**

---

### **1️⃣ SSA Dominance**

> Ops can only use values that are in scope, i.e. visible according to SSA dominance, nesting, and semantic restrictions imposed by enclosing operations. Values are visible within a CFG if they obey standard SSA dominance relationships, where control is guaranteed to pass through a definition before reaching a use.

**An Op can only use values that are in scope, which means they must follow SSA dominance, nesting rules, and semantic restrictions imposed by enclosing operations. Within a control flow graph (CFG), a value is visible if it obeys the standard SSA dominance, meaning control must pass through the definition of the value before reaching its use.**

📌 **Key Points**

- **MLIR follows the SSA (Single Static Assignment) rule**, meaning variables (Values) **can only be assigned once** and **must be defined before use**.
- **Dominance ensures that "all variable uses must happen after their definition"**, or it will violate the SSA rule.
- **This mechanism guarantees that the variable scope and control flow correctness are maintained during transformations in different passes.**

📌 **Example**

```
^bb0:
  %a = "some.op"() : () -> i32
  "some.other_op"(%a) : (i32) -> ()
```

- Here, **`%a`** is **defined** in `^bb0` before being used by `"some.other_op"`, which complies with **SSA dominance**.
- If `"some.other_op"` were to reference `%a` **before** `"some.op"`, it would violate **SSA rules**.

---

### **2️⃣ Region-based Visibility**

> Region-based visibility is defined based on simple nesting of regions: if the operand to an Op is outside the current region, then it must be defined lexically above and outside the region of the use.

**Visibility of variables within a Region is determined by the nesting structure: if a variable used by an Op is outside the current region, its definition must be lexically above and outside the region.**

📌 **Key Points**

- **Each Op in MLIR can have its own Regions**, and **variable visibility** is controlled by the nesting structure.
- **Operations within a Region can access variables defined outside it**, but cannot see variables defined inside other Regions.
- **This allows MLIR to support lexical scoping**, e.g., an `affine.for` loop can access external variables, but internal variables won't affect the outer scope.

📌 **Example**

```
%a = "some.op"() : () -> i32
"affine.for" () ({
  ^bb0:
    "some.inner_op"(%a) : (i32) -> ()
}) : () -> ()
```

- In this example, **`"some.inner_op"`** inside the `affine.for` loop can access **`%a`**, because `%a` is defined externally.
- However, if `"some.inner_op"` defines a new variable `%b`, it **cannot be used externally**.

---

### **3️⃣ Isolated from Above**

> MLIR also allows operations to be defined as isolated from above, indicating that the operation is a scope barrier—e.g. the “std.func” Op defines a function, and it is not valid for operations within the function to refer to values defined outside the function.

**MLIR allows certain operations to be marked as "isolated from above," indicating that the operation acts as a scope barrier. For example, the `std.func` Op defines a function, and operations within that function cannot refer to values defined outside of it.**

📌 **Key Points**

- Operations that are "isolated from above" act as a **scope barrier**, preventing internal access to external variables.
- **For example, `std.func` (functions) are "isolated from above"** operations, meaning internal operations **cannot access variables outside the function**, ensuring clear variable scope.
- **This allows MLIR to support parallel processing**, as there is no use-definition chain across scopes.

📌 **Example**

```
module {
  %a = "some.op"() : () -> i32
  func @my_function() {
    "some.inner_op"(%a) : (i32) -> ()  // ❌ This is wrong!
  }
}
```

- Here, **`func @my_function`** is an "isolated from above" operation, so it **cannot access `%a`**, which is defined externally.
- **Correct Code:**

```
func @my_function(%arg0: i32) {
  "some.inner_op"(%arg0) : (i32) -> ()
}
```

- In this case, **`%arg0` is a function argument**, so it is valid.

---

### **🔹 MLIR Advantages Over LLVM IR**

| Feature | MLIR | LLVM IR |
| --- | --- | --- |
| **SSA Dominance** | ✅ (Built-in SSA) | ✅ (Built-in SSA) |
| **Region-based Visibility** | ✅ (Nested Regions Control Scope) | ❌ (Flat CFG) |
| **Scope Barrier** | ✅ (`Isolated from Above`) | ❌ (Function variables need manual management) |
| **Support for Parallel Compilation** | ✅ (`Isolated from Above` allows avoiding Use-Def dependency) | ❌ (Needs extra analysis) |

- **MLIR uses dominance and region-based visibility to manage variable scope**, making it **more flexible and intuitive** than LLVM IR.
- **MLIR’s "Isolated from Above" mechanism ensures scope isolation** and allows **parallel compilation**.

---

### **🔹 Summary**

1. **SSA Dominance**: All variables **must be defined before use**, ensuring SSA correctness.
2. **Region-based Visibility**: **Regions can access external variables**, but internal variables cannot affect the external scope.
3. **Isolated from Above**: Certain operations (e.g., `std.func`) act as **scope barriers**, ensuring that internal variables don't rely on external ones.
4. **Parallel Processing Advantage**: The `Isolated from Above` feature **enables parallel compilation**, improving performance.

## **4.Symbols and Symbol Tables**

These features make **MLIR more suitable for DSLs (Domain-Specific Languages), high-level language translation, and applications in AI/mathematical computations**! 🚀

This section discusses the **"Symbols"** and **"Symbol Tables"** in **MLIR (Multi-Level Intermediate Representation)**. These concepts provide a **standardized way to manage named IR objects**, such as **functions, global variables, modules**, and solve the problem of **how these objects are referenced and organized**.

---

## **🔹 Terminology and Corresponding Original Text**

---

### **1️⃣ Symbol Table**

> Ops can have a symbol table attached. This table is a standardized way of associating names, represented as strings, to IR objects, called symbols.

**Certain operations (Ops) can attach a symbol table, which is a standardized way of associating names (strings) with IR objects, known as symbols.**

📌 **Key Points**

- **The Symbol Table is a standard mechanism in MLIR to manage named IR objects**, such as functions, global variables, or modules.
- **A symbol's name is a string**, similar to a **variable name or function name** in a programming language.
- **A symbol is an IR object**, and its specific semantics are defined by the operation.

---

### **2️⃣ Use of Symbols**

> The IR does not prescribe what symbols are used for, leaving it up to the Op definition. Symbols are most useful for named entities that do not obey SSA: they cannot be redefined within the same table, but they can be used prior to their definition.

**MLIR does not enforce a specific use for "symbols"; their use is determined by the definition of the specific operation. Symbols are most useful for "named entities that do not obey SSA". They cannot be redefined within the same table but can be used before their definition (Forward Reference).**

📌 **Key Points**

- **Symbols are different from MLIR's "Values"** in that they do not follow SSA rules. That is:
    - Symbols **can be used before they are defined** (Forward Reference).
    - **But a symbol name cannot be redefined within the same symbol table** (to avoid name conflicts).
- **This allows MLIR to manage entities like functions and global variables**, which do not fit SSA (for example, recursive function calls are hard to handle under SSA rules).
- **In other words, a symbol table can be viewed as a "namespace"** in MLIR IR, used to manage objects like functions and global variables that do not fit under SSA management.

📌 **Example**

```
module {
  func @foo() {
    call @bar() : () -> ()
  }
  func @bar() {
    call @foo() : () -> ()
  }
}
```

Here:

- The functions **`@foo`** and **`@bar`** call each other, but they are used before being defined, which **cannot be done under SSA rules**.
- **The symbol table allows this "use before definition" (Forward Reference) mechanism, solving the recursive function problem**.

---

### **3️⃣ Nested Symbol Tables**

> Symbol tables can be nested if an Op with a symbol table attached has associated regions containing similar Ops.

**If an operation with a symbol table has regions that contain other operations that also use symbol tables, then nested symbol tables can be formed.**

📌 **Key Points**

- **Some operations** (such as `module` or `func`) **can have symbol tables**, allowing them to contain named IR objects.
- **When these operations contain other operations with symbol tables**, a **nested symbol table** is formed.
- **This structure allows for more flexible symbol namespace management**, similar to the **"scope"** mechanism in programming languages.

📌 **Example**

```
module {
  func @foo() {
    "my.op"() : () -> ()
  }
  module @nested {
    func @bar() {
      "my.op"() : () -> ()
    }
  }
}
```

Here:

- The outer **`module`** contains a function **`@foo`**, which can access symbols within the module (such as `@foo`).
- The inner **`module @nested`** contains a function **`@bar`**, but its namespace is **independent of the outer module**, meaning the **inner `@bar` does not affect the outer `@foo`**.

---

### **4️⃣ Referencing Symbols**

> MLIR provides a mechanism to reference symbols from an Op, including nested symbols.

**MLIR provides a mechanism for operations to reference symbols, including nested symbols.**

📌 **Key Points**

- **Operations can reference other symbols** by their **symbol name**, such as calling a function (`call @foo()`).
- **If a symbol is inside a nested symbol table**, it can be referenced using its **nested name**.
- **This mechanism ensures namespace consistency and prevents name clashes**.

📌 **Example**

```
module {
  module @nested {
    func @bar() {
      "my.op"() : () -> ()
    }
  }
  func @foo() {
    call @nested::@bar() : () -> ()
  }
}
```

Here:

- The **inner `@nested` module's `@bar`** cannot be directly called from within `@foo` because they reside in different symbol tables.
- However, **`@foo` can still reference `@bar` correctly using `@nested::@bar()`**.

---

### **MLIR Symbol Tables vs. LLVM Symbol Tables**

| Feature                      | MLIR                          | LLVM IR                  |
|------------------------------|-------------------------------|--------------------------|
| **Symbol Mechanism**          | ✅ (`Symbol Table`)            | ✅ (`Global Symbol Table`)|
| **Supports Forward Reference**| ✅ (Allows use before definition) | ❌ (Needs additional handling)|
| **Supports Nested Symbol Tables**| ✅ (Manages multi-level namespaces) | ❌ (Flat structure)       |
| **Supports Recursive Function Calls** | ✅ (Managed via symbols)   | ✅ (Managed via function pointers) |
  
- **MLIR's symbol table mechanism is more flexible than LLVM's** because it supports nested structures, allowing operations at different scopes to use their own symbol tables.
- **MLIR's symbol table allows "use before definition"**, making recursive functions and similar features easier to manage, while LLVM requires manual handling of function pointers to achieve this.

---

### **Summary**

1. **Symbol Tables** are the standard way MLIR manages named IR objects (functions, global variables, modules).
2. **Symbols** do not follow SSA rules, allowing **"use before definition" (Forward Reference)**, making them suitable for recursive functions and global variables.
3. **Nested Symbol Tables** allow MLIR to have separate symbol management mechanisms within different scopes, similar to namespaces.
4. **MLIR provides a standard mechanism to reference symbols**, ensuring namespace consistency and avoiding name clashes.

## **5.Dialect**

This section mainly discusses the **Dialect mechanism in MLIR (Multi-Level Intermediate Representation)**, which provides a **scalable framework that allows users to define their own Operations, Types, and Attributes**. It also enables **different Dialects to be mixed together, allowing for more flexible IR design**.

---

## **🔹 Terminology Explanation and Corresponding Original Text**

---

### **1️⃣ Dialect**

> MLIR manages extensibility using Dialects, which provide a logical grouping of Ops, attributes, and types under a unique namespace.

**MLIR manages extensibility through "Dialect", which provides a logical grouping mechanism to organize Operations (Ops), Attributes, and Types within an independent namespace.**

📌 **Key Points**

- **Dialect is the foundation of MLIR's extensibility**, allowing developers to **customize Operations, Types, and Attributes** for specific application domains (e.g., DSLs, deep learning, hardware simulation, etc.).
- **Each Dialect has its own "namespace"**, preventing name collisions. For example:
    - `std.add` (addition operation in the standard dialect `std`)
    - `affine.for` (loop operation in the affine dialect)
    - `vector.shuffle` (shuffle operation in the vector dialect)
- **A Dialect itself does not introduce new semantics**; it simply provides **a way to organize Ops, Types, and Attributes**.

📌 **Example**

```
%result = affine.apply affine_map<(d0) -> (d0 + 1)>(%x)
```

Here:

- `affine.apply` belongs to the **`affine` dialect**, representing an **affine operation** (Affine Transformation).
- `%x` is the input variable, and `affine_map<(d0) -> (d0 + 1)>` represents a **mathematical transformation** (d0 + 1).
- This is an **operation specific to the Affine Dialect** in MLIR.

---

### **2️⃣ Dialect Namespace**

> The dialect namespace appears as a dot-separated prefix in the opcode, e.g., Figure 4 uses affine and std dialects.

**The namespace of a Dialect appears as a dot-separated prefix in the operation code (OpCode), such as `affine.for` (Affine Dialect) and `std.add` (Standard Dialect).**

📌 **Key Points**

- **Each Op belongs to a specific Dialect** and is named in the format `dialect.op`. For example:
    - `std.add` → **addition in the standard (`std`) dialect**
    - `affine.for` → **loop in the affine (`affine`) dialect**
    - `vector.extract` → **extract in the vector (`vector`) dialect**
- **This naming convention helps avoid name collisions**, because `std.add` and `custom.add` are **different operations**.

📌 **Example**

```
%sum = std.addf %a, %b : f32
```

Here:

- `std.addf` belongs to the **standard (std) dialect**, representing **floating-point addition**.
- `: f32` indicates that **this operation operates on the `f32` type**.

---

### **3️⃣ Dialect Makes MLIR Like a "Modular Library"**

> The separation of Ops, types, and attributes into dialects is conceptual and is akin to designing a set of modular libraries.

**The concept of Dialects is similar to a "modular library", where each Dialect corresponds to a set of Ops, Types, and Attributes for specific purposes, making MLIR's IR structure clearer.**

📌 **Key Points**

- **Different Dialects are like different libraries**, each responsible for different functions. For example:
    - **`affine` Dialect** → suitable for handling **loops with statically known bounds (Loop Transformation, Loop Unrolling)**
    - **`vector` Dialect** → suitable for handling **vector computations (SIMD, AVX, SVE)**
    - **`gpu` Dialect** → suitable for handling **GPU operations (CUDA, OpenCL, ROCm)**
- **Developers can create custom Dialects** to provide optimization and operations for specific domains.

---

### **4️⃣ Mixing Dialects**

> Ops from different dialects can coexist at any level of the IR at any time, they can use types defined in different dialects, etc.

**Operations from different Dialects can coexist in the IR and use each other's Types and Attributes, offering great flexibility.**

📌 **Key Points**

- **Operations from different Dialects can coexist at the same level of the IR**, for example:
    - `std` dialect's addition (`std.addf`)
    - `vector` dialect's vector operation (`vector.extract`)
    - `gpu` dialect's GPU operation (`gpu.launch`)
- **Ops can use Types defined in different Dialects**, for example:
    - `gpu.launch` (GPU dialect) **can operate on Types from the `vector` dialect**
    - `affine.for` (Affine dialect) **can use operations from the `std` dialect**
- **This design provides great extensibility**, allowing developers from different domains to create their own Dialects and mix them.

📌 **Example**

```
%vec = vector.broadcast %scalar : f32 to vector<4xf32>
%sum = std.addf %vec, %vec : vector<4xf32>
```

Here:

- **The first line (`vector.broadcast`) belongs to the `vector` dialect**, representing broadcasting a scalar `f32` to a `vector<4xf32>`.
- **The second line (`std.addf`) belongs to the `std` dialect**, representing adding two `vector<4xf32>`s.
- **This demonstrates that `std` and `vector` dialects can be mixed**.

---

### **5️⃣ Progressive Lowering**

> MLIR explicitly supports a mix of dialects to enable progressive lowering.

**MLIR allows mixing different Dialects, making "progressive lowering" more flexible.**

📌 **Key Points**

- **Lowering refers to the process of converting higher-level IR into lower-level IR**, until finally producing LLVM IR or machine code.
- **Dialects allow this process to be done in stages**:
    1. **High-level DSL → High-level IR (Custom Dialect)**, e.g., TensorFlow Dialect (`tf`) or PyTorch Dialect (`torch`).
    2. **High-level IR → Mid-level IR (Standard MLIR Dialects)**, e.g., `linalg`, `affine`, `vector`.
    3. **Mid-level IR → Low-level IR (Closer to LLVM IR)**, e.g., `llvm` Dialect, `gpu` Dialect.
    4. **Low-level IR → LLVM IR or Machine Code**, converted to LLVM IR (`llvm` Dialect) or SPIR-V (for Vulkan GPUs).
- **This makes MLIR highly flexible and capable of supporting various computational platforms**.

---

## **🔹 Summary**

1. **Dialect is MLIR's extensibility mechanism, where each Dialect provides a set of specific Operations, Types, and Attributes.**
2. **Each Dialect has an independent "namespace" to avoid name collisions between different Dialects' Ops.**
3. **Different Dialects can coexist and reference each other's Types and Attributes, increasing flexibility.**
4. **MLIR supports "progressive lowering", with IR being lowered step by step through different Dialects until it becomes LLVM IR or machine code.**



## **6. Type System and Standard Types**

This section mainly explains the **MLIR Type System**, covering how **types (Type)** are defined, the **extensibility of MLIR types**, **standard types**, and the **theoretical foundation of MLIR's type system**.

---

## **🔹 Terminology and Corresponding Original Text**

---

### **1️⃣ MLIR Type System**

> Each value in MLIR has a type, which is specified in the Op that produces the value or in the block that defines the value as an argument.
> 

**Each value in MLIR has a type, which can be specified by the Operation that produces the value, or by the block that defines the value as a parameter.**

📌 **Key Points**

- **MLIR is a statically typed (Static Typing) system**, where every **Value must have a defined Type**.
- **Types for Values come from two sources**:
    1. **Generated by an Op (Operation Output)**  
       For example, `std.addf %a, %b : f32` — the result generated by `std.addf` is of type `f32`.
    2. **Defined by Block Parameters**  
       Parameters defined within a block have explicit types, for example:

        ```
        ^bb0(%arg0: i32)  // %arg0 is explicitly defined as type i32
        ```

- **Type definitions convey the semantics (compile-time semantics)**, and are checked during compilation.

---

### **2️⃣ Extensibility of the Type System (User-Extensible Type System)**

> The type system in MLIR is user-extensible, and may refer to existing foreign type systems (e.g. an llvm::Type or a clang::Type).
> 

**MLIR's type system is extensible, allowing the reference to external type systems, such as LLVM's `llvm::Type` or Clang's `clang::Type`.**

📌 **Key Points**

- **MLIR allows users to define custom Types**, enabling the creation of new types for specific domains (e.g., AI, quantum computing, image processing).
- **MLIR also supports integration with external Types**, such as LLVM IR's Type, which allows MLIR to produce output compatible with LLVM IR.
- **This makes MLIR a flexible IR that can adapt to different computational domains**.

📌 **Example**

```
llvm.func @my_function(%arg0: !llvm.ptr<i8>) -> !llvm.i32
```

Here, `!llvm.ptr<i8>` represents an **LLVM Pointer Type**, and `!llvm.i32` represents an **LLVM 32-bit Integer Type**, illustrating **MLIR's ability to reference LLVM Types**.

---

### **3️⃣ Strict Type Equality Checking in MLIR**

> MLIR enforces strict type equality checking and does not provide type conversion rules.
> 

**MLIR enforces strict type equality checking and does not allow automatic type conversions.**

📌 **Key Points**

- **MLIR does not have implicit type conversions**, unlike C/C++/Python.
- **Types must match exactly**, otherwise an error will occur.
- **If you need to convert types, you must use explicit Type Casting operations**, such as `std.cast`.

📌 **Example**

```
%result = std.addf %a, %b : f32  // This is valid
%wrong = std.addf %a, %b : f64  // Invalid, types don't match
```

If `a` and `b` are not of type `f32`, MLIR will not automatically convert them, and you'll need to manually cast them:

```
%converted = std.cast %a : f64 to f32
%result = std.addf %converted, %b : f32
```

This ensures that types match correctly.

---

### **4️⃣ Operation Type Syntax**

> Ops list their inputs and result types using trailing function-like syntax.
> 

**MLIR Operations use function-like syntax to list input types and output types.**

📌 **Key Points**

- **MLIR Ops use function-style syntax to specify input and output types**.
- Syntax: `input types -> output types`.
- **This syntax clearly describes the type behavior of an Operation**.

📌 **Example**

```
func @compute(%arg0: i32, %arg1: f32) -> f32 {
  %result = std.addf %arg1, %arg1 : f32
  return %result : f32
}
```

Here:

- `func @compute(%arg0: i32, %arg1: f32) -> f32` represents a function `compute`, with input types `i32` and `f32`, and output type `f32`.
- `%result = std.addf %arg1, %arg1 : f32` indicates that the `std.addf` operation **takes and returns `f32`**.

---

### **5️⃣ MLIR Type Theory**

> MLIR only supports non-dependent types, including trivial, parametric, function, sum, and product types.
> 

**MLIR only supports "non-dependent types", including trivial types, parametric types, function types, and sum/product types.**

📌 **Key Points**

- **MLIR's type system is based on Type Theory**, and it primarily supports **non-dependent types**.
- **Main types include**:
    - **Trivial Type** → Basic types, such as `i32`, `f32`.
    - **Parametric Type** → For example, `vector<4xf32>`.
    - **Function Type** → For example, `(i32, f32) -> f32`.
    - **Sum Type** → For example, `A | B`, similar to `std::variant` in C++.
    - **Product Type** → For example, `(i32, f32, f64)`, similar to `std::tuple` in C++.

---

### **6️⃣ Standard Types in MLIR**

> MLIR provides a standardized set of commonly used types, including arbitrary precision integers, standard floating point types, and simple common containers—tuples, multi-dimensional vectors, and tensors.
> 

**MLIR provides a set of standard types, including**:

1. **Integer Types** → `i1`, `i8`, `i32`, `i64`, and even extendable to `i128`.
2. **Floating-Point Types** → `f16`, `f32`, `f64`.
3. **Tuple Types** → `(i32, f32)`, similar to `std::tuple`.
4. **Vector Types** → `vector<4xf32>`, suitable for SIMD operations.
5. **Tensor Types** → `tensor<4x4xf32>`, suitable for AI/machine learning.

📌 **Example**

```
%vec = vector.broadcast %scalar : f32 to vector<4xf32>
%tensor = linalg.fill %tensor_dest, %value : tensor<4x4xf32>
```

Here:

- `vector<4xf32>` is **suitable for SIMD operations**.
- `tensor<4x4xf32>` is **suitable for AI computations**.

---

## **🔹 Summary**

1. **MLIR's type system is static, and every Value has an explicit Type.**
2. **MLIR's type system is extensible and can reference types from LLVM or Clang.**
3. **MLIR enforces strict type checking and does not allow implicit type conversions.**
4. **Ops use function-style syntax (`input -> output`) to denote type relationships.**
5. **MLIR's type system is based on Type Theory and supports non-dependent types.**
6. **MLIR provides standard types, including integers, floating points, vectors, and tensors.**

## **7.Functions and Modules**

This section primarily explains **Functions and Modules in MLIR**, describing their structure, how they are represented in MLIR, and their control flow behavior.


## **🔹 Terminology and Corresponding Original Text**

---

### **1️⃣ Functions and Modules in MLIR**

> Similarly to conventional IRs, MLIR is usually structured into functions and modules.
> 

**MLIR is structured similarly to conventional Intermediate Representations (IRs), typically consisting of "Functions" and "Modules."**

📌 **Key Points**

- In most compiler IRs (such as LLVM IR), **the code is organized into functions and modules**.
- **MLIR follows this design, but functions and modules are not special concepts in MLIR**. Instead, they are implemented via Operations from the `builtin` dialect.
- **A module can contain functions, global variables, and compiler metadata**.

---

### **2️⃣ Module**

> A module is an Op with a single region containing a single block, and terminated by a dummy Op that does not transfer the control flow.
> 

**A module is an Operation with a single Region, which contains a single Block, and ends with a dummy Op that does not affect control flow.**

📌 **Key Points**

- **`module` is the highest-level structure in MLIR, analogous to `Module` in LLVM IR**.
- **The `module` contains a single `region`, which in turn contains a single `block` that may contain functions, global variables, or compiler metadata**.
- **A module is a symbol and can be referenced**, meaning it can be accessed across Passes or Dialects.

📌 **Example**

```
module {
  func @add(%a: i32, %b: i32) -> i32 {
    %sum = std.addi %a, %b : i32
    return %sum : i32
  }
}
```

Here:

- `module` is the **top-level Operation**, containing a Region with a Block.
- `func @add` is a **function (Function Op)** inside the `module`.

---

### **3️⃣ Function**

> A function is an Op with a single region, with arguments corresponding to function arguments.
> 

**A function is an Operation with a single Region, and the arguments of the Region correspond to the function's parameters.**

📌 **Key Points**

- **A function (`func`) is also an Op**, and its `region` contains the function's implementation.
- **Function parameters directly correspond to the parameters of the `block` inside the `region`**.
- **The function's name is a symbol**, and can be called (referenced) elsewhere.

📌 **Example**

```
func @multiply(%arg0: i32, %arg1: i32) -> i32 {
  %result = std.muli %arg0, %arg1 : i32
  return %result : i32
}
```

Here:

- `func @multiply` defines a function named `multiply`.
- `(%arg0: i32, %arg1: i32)` are the function parameters.
- `return %result : i32` represents the function's return value.

---

### **4️⃣ Function Control Flow**

> The control flow is transferred into a function using a function call Op.
> 

**The control flow is transferred into a function using the `call` Operation.**

📌 **Key Points**

- **Function calls (`Call`) are handled by the `call` Op**, which transfers control flow to the specified function.
- **The function's internal control flow follows standard Control Flow Graph (CFG) rules**.
- **A Block within a function ends with a Terminator Op (such as `return`)**, which returns control flow back to the calling location.

📌 **Example**

```
func @square(%x: i32) -> i32 {
  %result = std.muli %x, %x : i32
  return %result : i32
}

func @main() -> i32 {
  %val = call @square(%c4_i32) : (i32) -> i32
  return %val : i32
}
```

Here:

- `func @square` defines a **square function**, taking `x` as input and returning `x * x`.
- `func @main` calls the `square` function using `call @square`, storing the result in `%val`.

---

### **5️⃣ `return` Terminates Function Execution**

> A "return" terminator does not have successors and instead terminates the region execution, transferring the control flow back to the call-site of the function.
> 

**The `return` terminator Op has no successors and instead terminates the function's Region execution, transferring the control flow back to the function's call-site.**

📌 **Key Points**

- **Blocks inside a function must end with a `return` terminator**.
- **The `return` has no successor blocks, and its Operand represents the function's return value**.
- **The `return` transfers control flow back to the `call` instruction's call-site**.

📌 **Example**

```
func @subtract(%a: i32, %b: i32) -> i32 {
  %result = std.subi %a, %b : i32
  return %result : i32
}
```

Here:

- `std.subi %a, %b : i32` calculates the subtraction of two numbers.
- `return %result : i32` **returns the result to the function's call-site**.

---

## **🔹 Summary**

1. **MLIR uses `module` to organize code, which is a top-level Op containing functions and other symbols**.
2. **A function (`func`) is also an Op, with a Region where the parameters correspond to the function's arguments**.
3. **A function's Block forms a control flow graph (CFG), and it ends with a `return` Op**.
4. **Function calls (`call`) are made via the `call` Op, transferring control flow to the function, and `return` sends control flow back**.
5. **Both modules and functions are symbols and can be referenced and manipulated in MLIR IR**.

