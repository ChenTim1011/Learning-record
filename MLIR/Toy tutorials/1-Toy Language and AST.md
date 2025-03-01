Reference:
[Chapter 1: Toy Language and AST](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/)

# **MLIR Toy Chapter 1**

## **Chapter 1: Toy Language and AST Explained**  

This chapter provides a detailed explanation of the **Toy language** and its **Abstract Syntax Tree (AST)**. It includes examples to help understand language design and syntax parsing.  

---

## **1. Introduction to the Toy Language**  

**Toy** is a simple, tensor-based language with the following features:  

- Supports **function definitions, mathematical operations, and output generation**.  
- Variables can only be **tensors (Rank ≤ 2)**, meaning they can be at most matrices.  
- **Single data type**: Only 64-bit floating-point numbers (`double` in C).  
- **Immutable variables**: All operations return new values instead of modifying existing ones.  
- **Automatic memory management**: No need for manual variable deallocation.  

### **Toy Language Example**  

```toy
def main() {
  # Define variable `a` with shape <2,3> and constant values.
  var a = [[1, 2, 3], [4, 5, 6]];

  # `b` has the same values as `a`, but written differently.
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # `transpose()` and `print()` are built-in functions:
  # 1. Transpose `a` and `b`
  # 2. Perform element-wise multiplication
  # 3. Print the result
  print(transpose(a) * transpose(b));
}
```

### **Language Features**  

1. **Type Inference**  
   - Toy supports static type checking and automatically infers types to reduce explicit type declarations.  
   - For example, `a` and `b` have their shapes inferred based on their initialization values.  

2. **Generic Functions**  
   - Functions accept tensors of **unspecified rank**, meaning the rank is known, but the exact dimensions are not.  
   - Functions undergo **specialization** when invoked with tensors of specific shapes.  

---

## **2. Functions and Type Inference**  

Toy allows function definitions and shape-based specialization.  

```toy
# Define a generic function that takes two tensors of unknown shape
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # Specialization for <2,3> tensors
  var c = multiply_transpose(a, b);

  # Same specialization as above
  var d = multiply_transpose(b, a);

  # New specialization triggered for <3,2> tensors
  var e = multiply_transpose(c, d);

  # Shape mismatch error: <2,3> and <3,2>
  var f = multiply_transpose(a, c);
}
```

---

## **3. Abstract Syntax Tree (AST)**  

The **Abstract Syntax Tree (AST)** represents the program's structure. Below is an example AST corresponding to the function above:  

### **AST Structure**  

```
Module:
  Function
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
```

This AST represents the function `multiply_transpose`, which includes:  

1. **Function Definition (`Proto`)**  
   - `multiply_transpose` is defined at line 4 with parameters `a` and `b`.  

2. **Function Body (`Block`)**  
   - **Return Statement**: The function returns the result of a multiplication operation (`BinOp: *`).  
   - **Function Calls (`Call 'transpose'`)**:  
     - `a` is transposed before multiplication.  
     - `b` is transposed before multiplication.  

---

## **4. Generating the AST**  

Toy provides a tool to generate the AST. Run the following command in the `examples/toy/Ch1/` directory:  

```sh
path/to/BUILD/bin/toyc-ch1 test/Examples/Toy/Ch1/ast.toy -emit=ast
```

This outputs the AST representation of the input program.

---

## **5. Lexer and Parser**  

Toy uses a **Recursive Descent Parser**, which consists of:  

- **Lexer (Lexical Analyzer)**  
  - Splits code into **tokens**.  
  - Defined in `examples/toy/Ch1/include/toy/Lexer.h`.  

- **Parser (Syntax Analyzer)**  
  - Builds the AST according to syntax rules.  
  - Defined in `examples/toy/Ch1/include/toy/Parser.h`.  

This is similar to the first two chapters of the **LLVM Kaleidoscope Tutorial**.

---

## **6. Next Step: Converting AST to MLIR**  

Now that we understand Toy's language and AST, the next chapter will cover **converting AST to MLIR (Multi-Level Intermediate Representation)**, a key step for compilation and optimization.  

---

### **Summary**  

1. **Toy Language** is a simple tensor-based language used for learning compiler techniques.  
2. **AST (Abstract Syntax Tree)** represents the program structure and can be viewed using `toyc-ch1 -emit=ast`.  
3. **Lexer/Parser** process the source code into an AST, similar to the LLVM **Kaleidoscope Tutorial**.  
4. **Next step**: Converting AST to **MLIR** for further optimization and compilation.  
