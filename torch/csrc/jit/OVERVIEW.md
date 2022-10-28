# JIT Technical Overview

The JIT can run and optimize PyTorch programs separate from the Python interpreter. This overview is organized into sections that go over different independent components:   
**JIT可以独立于Python解释器运行和优化PyTorch程序**

1. Core Program Representation -  The JIT executes TorchScript, a subset of python. This section describes how TorchScript programs are represented in the JIT, and serves as the interchange format between components of the JIT.  
JIT执行TorchScript，一个python子集，本节描述了在JIT中TorchScript programs如何被表示，并且作为JIT组件之间的交换格式
2. Generating Programs - TorchScript programs can be created either through tracing Python code or through directly writing TorchScript. This section describes how Models are created from these frontends.  
TorchScript programs可以通过trace或者TorchScript生成，这节描述了如何从这些前端创建Models

3. Executing Programs - Once created, TorchScript models are optimized and run. Since this is a just-in-time compiler, programs are optimized as they are executed, so this section describes both how programs are optimized and how they get run.  
Excuting Programs一旦被创建，TorchScript models就可以被优化和运行。因为这是一个即时编译器，程序在执行时就会优化，这节描述了程序如何被优化和运行 

4. Saving Programs - TorchScript is often created in Python and then used from C++. This section describes how the save and load process works.   
TorchScript在python侧创建，在C++侧使用，这节描述了保存和加载过程的工作原理  

5. Python Bindings - TorchScript code is normally created and used from Python, so this section describes how the Python components interact with the code in this directory.  
TorchScript code在python侧创建和使用，这节描述了python组件和code如何交互

For concepts that are actual classes in the JIT, we use capitalized words, e.g. Graph or Value.

Sections start with a reference to the source file where the code related to the section resides.

## Table of Contents

- [JIT Technical Overview](#jit-technical-overview)
  - [Table of Contents](#table-of-contents)
- [Core Program Representation](#core-program-representation)
  - [Modules](#modules)
  - [Parameters](#parameters)
  - [Method](#method)
  - [FunctionSchema](#functionschema)
  - [Graph](#graph)
  - [Node](#node)
  - [Block](#block)
    - [If](#if)
    - [Loops](#loops)
  - [Value](#value)
  - [Type](#type)
- [Generating Programs](#generating-programs)
  - [Tracer](#tracer)
  - [Script](#script)
  - [Tree](#tree)
  - [Tree Views](#tree-views)
  - [frontend.py](#frontendpy)
  - [Lexer](#lexer)
  - [Tokens](#tokens)
  - [Parser](#parser)
  - [Compiler](#compiler)
  - [SugaredValue](#sugaredvalue)
  - [Resolver](#resolver)
  - [Environment](#environment)
  - [SSA Conversion](#convert_to_ssa)
  - [Python-Compiler Interaction](#python-compiler-interaction)
- [Executing Programs](#executing-programs)
  - [Evaluation Semantics](#evaluation-semantics)
  - [IValue](#ivalue)
  - [Operation](#operation)
  - [Operator](#operator)
  - [Interpreter](#interpreter)
  - [Graph Executor](#graph-executor)
  - [JIT Logging](#jit-logging)
  - [DifferentiableGraphOp](#differentiablegraphop)
  - [Interpreter](#interpreter-1)
  - [FusionGroup](#fusiongroup)
  - [Handling Mutability](#handling-mutability)
    - [Aliasing and mutation in the PyTorch API](#aliasing-and-mutation-in-the-pytorch-api)
    - [Aliasing and mutation annotations in FunctionSchema](#aliasing-and-mutation-annotations-in-functionschema)
    - [Alias Analysis in the IR](#alias-analysis-in-the-ir)
    - [Writing optimization passes with AliasDb](#writing-optimization-passes-with-aliasdb)
- [Profiling Programs](#profiling-programs)
- [Saving Programs](#saving-programs)
- [Testing Programs](#testing-programs)
  - [Test Autodiff](#test-autodiff)
  - [Python Printer](#python-printer)
- [Python Bindings](#python-bindings)


# Core Program Representation

## Modules ##

[api/module.h](api/module.h)

At the top level, all TorchScript programs are represented as a Module. Modules contain:  
所有的TorchScript programs都被表示为一个Module，Modules包含：

* named Parameters - tensors used in training such as `weight` or `bias`  
训练中使用的张量，如weight、bias

* named Buffers - tensors that are part of the training state of a module but do not appear in module.parameters() and do not participate in gradient descent.  
作为模块训练状态的部分张量，它不出现在module.parameters()，也不参与剃度下降

* named sub-Modules - used for code organization.  
用于code组织

* named Attributes - all other attributes that are not in the above three categories. Typically used for configuration and are not saved/restored in the modules `state_dict`.  
不属于上述三类的其它属性，通常用于配置，在modules state_dict不会保存和恢复

* named Methods - functions that can be run on the module such as `forward`  
可以运行在module上的函数，像forward

This mirrors the `nn.Module` objects used in Python. All TorchScript code is a member of some module. This includes pure functions such as those created by annotating a Python function with `@torch.jit.script`, which are represented internally as a Module that has a single method `forward` that contains the implementation of the function.  
所有的TorchScript code都是某个module的成员。包括pure functions，像用@torch.jit.script注释python函数创建的函数，它们被表示为有一个单一方法forward的Module

## Parameters ##

[api/module.h](api/module.h)

Modules contain Parameter objects, which simply hold a "slot" where a Tensor can be placed. These tensors are accessible by the Methods of the Module or the parent Module.  
Modules包含Parameter对象，它用一个槽来存放张量，这些张量可以通过Module或者父类Module的方法进行访问

## Method ##

[api/module.h](api/module.h)

A Method is a piece of TorchScript code that takes a number of arguments and produces an output value. Methods have several subcomponents. A FunctionSchema describes the types and names of the input arguments and return value. A list of `member_inputs` describes which Parameters are accessed by the method (this is blank for pure functions). A Graph object describes the actual code inside the method. The Method also maintains a GraphExecutor which is used to actually execute the Graph that defines the method.  
Method是一段TorchScript code，它接受多个参数并产生一个输出，Methods有几个子组件，FunctionSchema 描述了返回值、输入参数的类型和名字。member_inputs列表描述了方法访问的Parameters（对于pure functions这是空的）。Graph对象描述了method内部的code。Method也维护了一个GraphExecutor用来执行Graph。

The Graph inside the Method is a pure function. The Parameters used by the Method are added as additional inputs to this graph before it is run. This allows the GraphExecutor to treat method inputs and method parameters the same for the purposes of optimization and execution, simplifying the process for executing programs.  
Method内部的Graph是一个pure函数。图运行之前Method使用的Parameters作为额外输入添加到图中，处于优化和执行的目的，允许GraphExecutor将method inputs和method parameters视为相同，从而简化了执行程序的过程

Methods also contain helper functions for inserting calls to the Method from other Method objects.
Methods还包含辅助函数，用于插入从其它Method 对象对Method的调用

## FunctionSchema ##

[aten/src/ATen/core/function_schema.h](../../../aten/src/ATen/core/function_schema.h)

Each Method has a FunctionSchema that describes the Types of the arguments and return values of a function. Operators (builtin primitives that are called by the Interpreter) also have FunctionSchema. FunctionSchema are analogous to a function _declaration_ in C++. They describe how to call the function but do not provide an implementation.  
每一个Method都有一个FunctionSchema，它描述了一个函数的参数和返回值的类型，Operators（解释器调用的内置源语）也有FunctionSchema，FunctionSchema类似于C++中的函数声明，它描述了如何调用函数，但不提供函数实现


## Graph ##

[ir.h](ir/ir.h)

Graphs are the root of the intermediate representation (IR) used to define the implementation of TorchScript functions. If you are familiar with [LLVM](llvm.org), they are analogous to an `llvm::Function` object. A Graph is composed of Nodes, Blocks, and Values. Nodes are instructions (e.g. do a matrix multiply). Nodes are organized into Blocks of sequentially executed Nodes. Each Node produces a list of output Values, and also consumes a list of input Values. As an example, a user may write the following TorchScript code:  
Graph是IR的根，IR用于定义TorchScript函数的实现，如果你熟悉LLVM，它们类似于llvm::Function对象，Graph是由Nodes、Blocks和Values组成，Nodes是指令（比如做矩阵乘法），顺序执行的Nodes被组织成Blocks，每个Node产生一个输出Values的列表，并且消费一个输入Values列表。例如，用户可以写下如下TorchScript代码


```python
@torch.jit.script
def f(a, b):
  c = a + b
  d = c * c
  e = torch.tanh(d * c)
  return d + (e + e)
```

The frontend, described later in this document will turn into a `Graph`:  
在前端，转换为一个Graph
```
graph(%0 : Double(2),
      %1 : Double(2)):
  %2 : int = prim::Constant[value=1]()
  %3 : Double(2) = aten::add(%0, %1, %2)
  %4 : Double(2) = aten::mul(%3, %3)
  %5 : Double(2) = aten::mul(%4, %3)
  %6 : Double(2) = aten::tanh(%5)
  %7 : Double(2) = aten::add(%6, %6, %2)
  %8 : Double(2) = aten::add(%5, %7, %2)
  return (%8)
```

This is the canonical textual representation of the IR. You should be able to easily find (almost all) of the elements we discussed above.  
这是一个IR的文本表示，你可以找到上面提到的元素  

- `graph` is the `Graph`
- `%x` are `Value`s
- `%x : Double(2)` is a type annotation of `Value` `%x` (see below for a list of supported types).  
Double是类型注释  

- `%x : T1, %y : T2 = namespace::name(%z, %w)` is a `Node` which represents the `namespace::name`operator (this name is usually referred to as the `Node`s _kind_). It takes `%z` and `%w` `Value`s as inputs, and returns two outputs (`%x`, `%y`) of types `T1` and `T2` respectively.  
`%x : T1, %y : T2 = namespace::name(%z, %w)` 是一个 `Node` ，代表`namespace::name`操作符（name就是`Node`的_kind_）。它接受
`%z` 和 `%w` `Value`作为输入，返回类型为 `T1` 和 `T2`的输出(`%x`, `%y`)   

Finally, nodes can have extra pieces of information assigned to them, which are called _attributes_. You can see that it's used in the `prim::Constant` node, which returns the `value` attribute when it's called. There's a fixed list of types you can attach:  
节点有额外的信息，称为_attributes_，你可以看到它用在`prim::Constant` 节点，当它调用时返回`value`属性，你可以附加一个固定的类型列表

- `int64_t`
- `double`
- `Tensor`
- `Graph` (useful for e.g. slicing subgraphs that are meant to be fused)
- `std::string`
- and lists of them (not nested)

Graphs in the JIT are in single-static assignment (SSA) form, meaning that each Value has precisely one defining Node that can be looked up directly from the Value (`Node* n = v.node()`).  
Graphs在JIT中是single-static assignment form（单静态分配方式），每个Value都有一个定义的节点（`Node* n = v.node()`）

**Ownership Model** Blocks, Nodes, and Values are _owned_ by the Graph they appear in and may only appear in a single Graph. This is enforced by assertions in the API. Creation and deletion of Block, Node, and Value objects is done via methods on Graph objects (e.g. `Graph::create`,  `Node::addOutput`, or `Node::addBlock`). This API also enforces certain consistency properties. For instance, `Node::destroy` removes a Node, but it is only valid to call this function if the Values produced by this node are no longer used, which can be accomplished using other functions such as `Value::replaceAllUsesWith`.  
Blocks、Nodes和Values是_owned_，并且出现在单个图表中。这是由API
中的断言强制的。创建和删除Block、Node和Value对象通过Graph对象方法（比如`Graph::create`,  `Node::addOutput`,`Node::addBlock`）。这个API还强制执行某些一致性属性。比如， `Node::destroy` 删除一个节点，只有该节点产生的Values不再使用，这个调用才是有效的，这可以由其他函数像`Value::replaceAllUsesWith`来完成

Because Graph owns all its Nodes, Values, and Blocks, these values are always passed around by raw pointer. Generally developers should not write code that holds Value, Node, or Block objects indefinitely without also holding a shared_ptr to their owning Graph.  
图拥有Nodes、Values和Blocks，这些值通过raw pointer传递，如果没有shared_ptr指向Graph，通常开发者不应该写无限期保存Value、Node和Block的代码

## Node ##

[ir.h](ir/ir.h)

A node represents a single built-in instruction such as a matrix multiply or a convolution. Each node has a `kind()` method that determines which builtin instruction the node represents. Different nodes (e.g. conv vs matrix-multiply) are represented using different kinds and not via subclassing of Node, as one would find in LLVM. A `kind()` is a `Symbol` object, which is just an "interned" string inside some namespace. Symbols can be created from strings, e.g. through `Symbol::fromQualString("aten::add")`, so there is not a closed set of `kind()` values that a Node might have. This design was chosen to allow the open registration of new operators and user-defined operators.  
Node表示一个指令，比如矩阵相乘或者卷积。每个node都有一个`kind()`方法，确定节点表示的内置指令。不同的节点（像conv和matrix-multiply）使用不同的类型表示，而不是像在LLVM中通过Node的子类来表示。`kind()` 是一个 `Symbol` 对象,只是某个命名空间中的内部字符串。Symbol可以通过string创建，比如，通过`Symbol::fromQualString("aten::add")`，因此没有一个Node拥有`kind()`值的封闭集。这种设计的目的是允许新操作符和用户定义的操作符进行注册。

>*Code in the JIT should always assume the universe of valid Node kinds is open and subject to be expanded.*  
JIT中的Code总是假定有效节点类型的范围是开放的，并且是可以扩展的

This reflects the reality of the PyTorch operator library where there are already several hundred valid operators.
这反映了PyToch operator库的实际情况，其中已经有数百个有效运算符

Nodes produce output Values and take input Values as arguments. For instance, a matrix-multiply will take two input tensors and produce one output tensor. Nodes can produce multiple outputs. For instance `prim::TupleUnpack` splits a tuple into its components, so it has a number of outputs equal to the number of members of the tuple. Though Nodes may have multiple outputs, the number of outputs is _statically known_ for each Node. Operations which may produce a dynamic amount of results, e.g. splitting a tensor into chunks of size 2, will be represented as an operator that results a list object.   
Nodes产生输出Values并且接受输入Values作为参数。例如，一个矩阵乘法接受两个输入张量并产生一个输出张量。节点可以产生多个输出，例如`prim::TupleUnpack`将一个tuple拆分成组件，因为它的输出数量等于tuple数量。尽管节点可能有多个输出，每个节点的输出数量是静态已知的。操作符可能产生动态结果，比如，划分一个张量到两个chunks，这将表示为结果为列表对象的一个操作符

Because Nodes are not subclassed per-operator, it is very easy to construct invalid Nodes, e.g. by forgetting an input or an output, or by passing Values of the wrong Type. To help avoid this, Graph provides the method (`Graph::insert`) for constructing Nodes that guarantees Nodes have the correct setup. This method uses the database of registered Operators and their FunctionSchema to construct Nodes using that schema.  
因为节点不是每个操作符的子类，非常容易构造无效节点，比如忘记输入或输出，或者传递错误的Values，为了避免这种情况，Graph提供了`Graph::insert`方法来构造节点，来保证节点有正确的设置，这个方法使用注册操作符的数据库和FuctionSchema来构造节点。

PyTorch IR supports function overloading  so the `kind()` of a node may correspond to multiple operators. For example, the kind `aten::add` has the following overloads (`Scalar` means `float` or `int` in this case):  
PyTorch IR 支持函数重载，因此一个节点的`kind()` 可能对应多个运算符。例如，`aten::add`有如下重载（在这种情况下`Scalar` 表示 `float` 或 `int`）：
- `aten::add(Tensor self, Tensor other) -> Tensor`
- `aten::add(Tensor self, Scalar other) -> Tensor`
- `aten::add(int self, int other) -> int`
- `aten::add(float self, float other) -> float`

For Nodes representing built-in Operators, the method `Node::schema` can also look up the FunctionSchema registered for that Operator.
对于内置Operators的Nodes，方法`Node::schema`查找为该运算符注册的FunctionSchema。

All of the strings correspond to different `FunctionSchema` objects. A `Node` can be queried for its schema using the `schema()` method (it will check the argument types, and will try to match one of the options for its `kind()`).
使用`schema()`方法查询`Node`的schema（它将检查参数类型并且将尝试匹配`kind()`的选项）

Note that the chosen overload is not shown in any way in the textual output. If you're unsure which function does a node resolve to, you might need to check the type annotations of its input values.  
注意，所选重载不会以任何方式显示在文本输出中，如果你不确定节点解析到哪个函数，你可能需要检查输入values的类型注释

Each node also has a set of attributes which are named integers, strings, floats, Tensors, and subgraphs, or lists of these types. These are used by special primitive operators to encode additional data in the Node. For instance `prim::Constant` defines a compile-time constant value. For Tensor constants, it will have a single Tensor attribute with the name `attr::value` which contains the value of the constant.  
每个节点还具有一组属性，为integers、strings、floats、Tensors 和 subgraphs，或者这些类型的列表。这些特殊的operators用来编码Node中的附加数据。例如`prim::Constant` 定义了一个编译期常量。对于常量张量，有一个`attr::value` 属性，其中包含常量的值。

Attributes are _rarely used_. Operators like convolution or matrix-multiply have no attributes and take of their arguments through the input list. This includes things that might be typically thought of as constants, like the stride of the convolution. In PyTorch, any of this information is potentially a dynamic property of the program so Nodes are always encoded in a way that allows these values to be dynamically determined. However, we recognize that many inputs are almost always constants, so we make it easy to quickly check if an input is constant and get its value with `c10::optional<IValue> Node::get(Symbol name)`, which returns an IValue (a concrete value for the input) in the case the node is constant and `nullopt` otherwise.  
Attributes很少使用，像卷积和矩阵乘法没有属性，他们通过输入列表接受参数。这通常包含一些常数的东西，比如卷积的步长。在PyTorch中，这些信息的任意一个都可能是程序的动态属性，因此Nodes允许以动态确定这些值的方式编码。然而，我们看到许多输入都是常量，我们可以快速的判断输入是常量，并使用`c10::optional<IValue> Node::get(Symbol name)`获取它的值，在节点为常量的情况下`c10::optional<IValue> Node::get(Symbol name)`返回一个IValue（输入的具体值），否则返回`nullopt`

## Block ##

[ir.h](ir/ir.h)

Nodes are organized into sequentially executed lists inside a Block. A Node is a member of precisely one Block. The Graph itself has a top-level `graph.block()`, and control-flow nodes (`prim::If` and `prim::Loop`) also have sub-blocks. While it is possible to design a Graph representation that does not have a sequential order for nodes (i.e. a sea-of-nodes representation), we find it is much easier to debug and understand Blocks when there is a specific canonical order for all of the nodes. This does not preclude optimization passes from changing the order when it would improve performance, and the interpreter is potentially allowed to execute the block out-of-order if the re-ordering preserves the semantics much like an out-of-order processor. Having the ordering ensure that graphs can always be easily printed, and that we can easily step through the execution of a graph.  
在Block中，Nodes被组织成顺序执行的列表。Node是一个Block的成员。Graph有一个顶层的`graph.block()`，控制流节点（`prim::If`和`prim::Loop`）也有sub-blocks。虽然可以设计一个没有nodes顺序的图（节点海），但是对所有节点有一个特定顺序，可以更容易的理解和调试Block。这并不排除优化passes提高性能时改变顺序。如果重新排序保留语义，就像乱序处理器一样，解释器可能也会乱序执行block。顺序可以确保图容易的打印出来，可以很容易的逐步执行一个图。

Values are Block-scoped. A Value is in scope for the remainder of the Block it is defined in, including in the sub-blocks of any Node defined after it. Values go out of scope at the end of the block in which they are defined.  
Values是在Block范围中的。Value在定义它的块的其余范围内，包括它之后定义的任意节点的sub-blocks。Values在定义它们的block末尾超出作用域。

When Nodes are inserted into a Graph, they are inserted at a special "insertion point" that is part of the state of the Graph. On construction, this will go to the end of the Graph.  
当Nodes插入到Graph，它们被插入到一个特殊的插入点，这是Graph状态的一部分。在构建时，这将转到图的末尾。

Each block has two dummy nodes that are not included in the list of nodes in the block. The `prim::Param` node represents the inputs to block and does have a `prev()` or `next()` node. The `prim::Return` node represents the outputs of a block.    
每个block有两个虚拟节点，它们没有包含在block的节点列表中。`prim::Param`代表block的输入，并且有一个`prev()`或者 `next()`节点。`prim::Return`节点代表block的输出。

The list of Nodes in a block is implemented as a circular linked list with the `prim::Return` Node serving as the beginning/end sentinel.  Inserting and deleting at arbitrary places is efficient. Developers may also encounter implementations inside of IR objects that use this fact (e.g. appending to a block is equivalent to putting the node before the `prim::Return` node).  
**block中的Nodes列表，使用循环链表实现**。`prim::Return`节点作为开始/结束标志。在任意位置插入和删除是有效的。开发者可能也会遇到IR对象的内部实现（例如，添加一个block相当于放一个节点在 `prim::Return`节点之前）

Iterators for the `nodes()` list are invalided when the current Node they point to is moved or deleted. Otherwise iterators remain valid.  
当迭代器指向的节点被移动或者删除，迭代器变得无效。

Block also contain a list of input and output values. The meaning of these values depends on where the block is used. For the Graph's top-level block, these are inputs and outputs to the Graph, and line up with the FunctionSchema associated with a Method.   
Block也包含输入和输出值的列表，这些值的含义取决于block的位置。对于Graph的顶层block，这些是Graph的输入和输出，与FunctionSchema关联的Method对齐。

**Control-flow** is represented with using sub-blocks rather than a control-flow graph representation. A `prim::If` has one block for the true branch and one block for the else.A `prim:Loop` has a block for the loop body (there is no condition block, instead the end of the loop body computes whether to re-enter the loop body). This representation ensures we have structured control-flow. This limitation makes a lot of optimizations easier and is true for the vast majority of networks. A Node can lookup what Block it is in, and a Block and can look up its parent (either the Node that has it as a subblock, or `nullptr` for the main Block).     
控制流使用sub-blocks表示而不是控制流图。 一个`prim::If`有一个true分支的block，还有一个else分支的block。一个`prim:Loop`有一个block用于循环体（没有条件block，而是在循环体的末尾计算是否重新进入循环体）。这种表示确保了我们有结构化的控制流。这种限制使得优化变得更容易，并且对大多数网络都是如此。一个Node可以查找它所在的Block，一个Block也可以查找它的parent（要么是subblock的Node，要么是mainblock的nullptr）。

### If ###
For if-statements (`prim::If`) the Blocks have no inputs, and the outputs are the new values of variables in the outer block whose values were altered in an if-statement.   
对于if，Block没有输入，输出是外部block变量的新值，外部block的值在if语句中被改变

Example IR for an if-statement looks like:    
if的IR示例：
```
%y_1, ..., %y_r = prim::If(%condition)
  block0():  # TRUE BRANCH, never takes arguments, has to return r outputs
    %t_1, ..., %t_k = some::node(%a_value_from_outer_block)
    -> (%t_1, ..., %t_r)
  block1():  # FALSE BRANCH, never takes arguments, has to return r outputs
    %f_1, ..., %f_m = some::node(%a_value_from_outer_block)
    -> (%f_1, ..., %f_r)
```

Values corresponding to `%y_1, ..., %y_r` will become either `%t_1, ..., %t_r`, or `%f_1, ..., %f_r` depending on the value of `%condition` at runtime (you can see that the node kind of acts as a Phi node in conventional SSA).  
依据运行时%condition的值，对应`%y_1, ..., %y_r`的值，将变为`%t_1, ..., %t_r`或者`%f_1, ..., %f_r`（node在传统SSA中扮演Phi node）

Here's an example translation of a Python program:

```python
def f(a, b, c):
    d = a + b
    if c:
        e = d + d
    else:
        e = b + d
    return e
```

```
graph(%a : Dynamic,
      %b : Dynamic,
      %c : Dynamic):
  %2 : int = prim::Constant[value=1]()
  %3 : Dynamic = aten::add(%a, %b, %2)
  %5 : Dynamic = prim::If(%c)
    block0():
      %6 : int = prim::Constant[value=1]()
      %7 : Dynamic = aten::add(%3, %3, %6)
      -> (%7)
    }
    block1():
      %8 : int = prim::Constant[value=1]()
      %9 : Dynamic = aten::add(%b, %3, %8)
      -> (%9)
  return (%5)
```

The outputs of the if-statement serve a role similar to "phi" nodes in traditional SSA control-flow graphs.  
if中的输出类似于在传统SSA控制流图中的phi nodes

### Loops ###
Loops are implemented with `prim::Loop` which covers both `while` and `for` loops. A valid instantiation of this node always looks like this:   
循环使用`prim::Loop`实现，它涵盖了`while`和`for` 。该节点的有效示例如下：

```
%y_1, ..., %y_r = prim::Loop(%max_trip_count, %initial_condition, %x_1, ..., %x_r)
  block0(%i, %a_1, ..., %a_r):
    %b_1, ..., %b_m = some::node(%a_value_from_outer_block, %a_1)
    %iter_condition = some::other_node(%a_2)
    -> (%iter_condition, %b_1, ..., %b_r)
```

The simplest way to explain the semantics is to consider this Python-like pseudo-code:  
解释语义最简单的方法是用类似Python的伪代码：  

```python
y_1, ..., y_r = x_1, ..., x_r
condition = initial_condition
i = 0
while condition and i < max_trip_count:
    a_1, ..., a_r = y_1, ..., y_r

    ############################################################
    # Actual body of the loop
    b_1, ..., b_m = some::node(a_value_from_outside_of_the_loop, a_1)
    iter_condition = some::node(a_2)
    ############################################################

    y_1, ..., y_r = b_1, ..., b_r
    condition = iter_condition
    i += 1
```

> Note that translations of `for` loops simply pass in a constant `true` for both `%initial_condition` and `%iter_condition`, while for `while` loops `%max_trip_count` is set to the largest value of `int64_t`, and `%i` is unused. Those patterns are recognized by our interpreter and optimized accordingly (e.g. `while` loops don't maintain the loop counter).    
注意，`for`只是简单传递一个常量`true` 给`%initial_condition`和`%iter_condition`，而对于 `while`，`%max_trip_count`被设置成`int64_t`的最大值，并且`%i`未被使用。这些模式被我们的解释器识别并进行相应的优化（例如 `while`循环不维护循环计数变量）

For example, this program:

```python
def f(x):
    z = x
    for i in range(x.size(0)):
        z = z * z
    return z
```

can be translated as:

```
graph(%z.1 : Dynamic):
  %3 : bool = prim::Constant[value=1]()
  %1 : int = prim::Constant[value=0]()
  %2 : int = aten::size(%z.1, %1)
  %z : Dynamic = prim::Loop(%2, %3, %z.1)
    block0(%i : int, %5 : Dynamic):
      %z.2 : Dynamic = aten::mul(%5, %5)
      -> (%3, %z.2)
  return (%z)
```

### With ###
With-statements are represented in two different ways. For most of the compilation and optimization process, they are represented as a pair of `prim::Enter` and `prim::Exit` nodes that wrap the nodes corresponding to the body of the with-statement. However, with-statements are temporarily represented for the duration of the `exit_transform` pass using a block-based representation in which a `prim::With` node is inserted after the `prim::Exit` node, all of the nodes between the `prim::Exit` and `prim::Enter` are moved into the first block of the `prim::With`, and the `prim::Exit` is moved into the second block of the `prim::With`. For example, this program:  
with用两种不同的方式表示，对于大多数编译和优化过程，它们被表示为一对`prim::Enter`和`prim::Exit`节点，它包含了with主体对应的节点。然而，`exit_transform`传递期间，使用基于block的表示临时表示with，在该表示中`prim::With`插入到`prim::Enter`节点后，`prim::Exit`和`prim::Enter` 之间的节点被移动到`prim::With`的第一个block中，`prim::Exit`被移动到`prim::With`的第二个block中，例如：

```
with c as mult:
  y = x + mult
```

can be translated as:

```
%2 : int = prim::Constant[value=1]()
%mult.1 : int = prim::Enter(%c.1)
%y.1 : Tensor = aten::add(%x.1, %mult.1, %2)
%11 : Tensor = prim::Exit(%c.1)
```

and will temporarily be transformed to:

```
%mult.1 : int = prim::Enter(%c.1)
= prim::With()
  block0():
    %y.1 : Tensor = aten::add(%x.1, %mult.1, %4)
    -> ()
  block1():
    %11 : Tensor = prim::Exit(%c.1)
    -> ()
```

for the duration of the `exit_transform` pass.  
对于`exit_transform`传递期间

## Value ##

[ir.h](ir/ir.h)

A Value represents data flowing through the operations in the program, e.g. the output of a matrix-multiply op. Value objects are always defined by a single Node (`v.node()`) due to single-static assignment form. For inputs to a Block/Graph, this node is a special `prim::Param` node that does not appear anywhere in the block's list of nodes. Value objects also have a Type (e.g. is it a tensor? a list? a tuple?) that provides a static guarantee that its value will be of that Type.  
Value表示在operations之间流动的数据，由于single-static分配模式，Value由单个节点（`v.node()`）定义，对于Block/Graph的输入，该节点是一个特殊的 `prim::Param`节点，它不会出现在block的节点列表中。Value对象也有一个类型（如tensor list tuple），它提供了一个静态保证，它的值是那个类型。

Value objects have methods on them to from the Value to its definition (`v.node()`) and to all of its uses `v.uses()`, which is a list of Nodes whose input list includes the value. Be careful when iterating over `v.uses()` while changing how `v` is used because each change to `v` will invalidate the `v.uses()` iterator.    
Vlaue对象具有从Value到其定义以及`v.uses()`的方法，它是一个节点列表，其输入列表包含value。当更改`v`的使用方式方式时，迭代`v.uses()`要小心，因为每一次对`v`的更改，都将会使`v.uses()`的迭代器失效。

Values are abstract representation of data in the program. When executing, the actual tensors, list, tuples, etc. are stored in IValues (_interpreter_ values), which are tagged unions of all possible values in TorchScript. In retrospect the name Value is a bit confusing because it seems like it should be the tagged union, but it originally came from analogy to `llvm::Value`, which serves the same purpose as `jit::Value`.    
在程序中，Values是数据的抽象表示。当执行时，**实际的tesors list tuples等存储在IValues（解释器values），在TorchScript中它是所有可能值的便签union**。回想起来Value的名字有点迷惑性，它看起来应该是标签union，它的名字来源于`llvm::Value`，和`jit::Value`具有相同的用途。

## Type ##

[aten/src/ATen/core/jit_type.h](../../../aten/src/ATen/core/jit_type.h)

TorchScript, unlike Python, is statically typed, so every Value has a Type associated with it, and every FunctionSchema has a list of argument types and a return type for a function. Type is the base class of a hierarchy of C++ objects that represent the built-in types of TorchScript. Types provide methods such as `Type::isSubtypeOf` that describe the typing relationships. Common type are:   
TorchScript和python不同，是一种静态类型，因此每一个Value都有一个Type与之关联，每一个FunctionSchema都有一个参数类型列表和一个函数返回类型。Type是C++对象层次的基类，它表示TorchScript的内置类型。Types提供了`Type::isSubtypeOf`来描述类型关系。常见的类型有：

* TensorType - a tensor with optionally refined information. It may optional know its device, type, requires_grad state, the number of dimensions.
  If it does know the number of dimensions it may optionally know the size of a particular dimension.  
  TensorType - 带有可选信息的tensor，可能知道device type requires_grad状态、维度数量，如果知道维度的数量，可能知道指定维度的大小  
* Tuples - e.g. Tuple[Tensor, Int]. Each member of the tuple is statically typed and the length of the tuple is statically known.  
Tuples - 例如 Tuple[Tensor, Int]，tuple的每个成员都是静态类型，并且tuple的长度是静态可知的  
* List[T] - e.g. List[Tensor]. Mutable lists of a particular type.  
List[T] - 例如List[Tensor]，指定类型的可变lists
* Optional[T] - e.g. Optional[Tensor], either the Tensor value or None.  
Optional[T] - 例如 Optional[Tensor]，Ternsor或者None  
* Dict[K, V] - e.g. Dict[String, Tensor], dictionaries  
Dict[K, V] - 例如Dict[String, Tensor]，字典  

If type S is a subtype of P, then we can substitute an IValue that has type S anywhere something of type P is expected. This means that all subtyping relationships also require the representation of the IValue for subtypes to be compatible with the representation for the base type.   
如果类型S是P的子类型，在需要类型P的地方可以使用具有类型S的IValue替换，这意味着所有子类型关系要求子类型的IValue表示要与基类的表示兼容。


# Generating Programs #

JIT programs are created using either the tracing frontend (`torch.jit.trace`) or the scripting frontend (`torch.jit.script`). In both cases, the result of these frontends is a complete Module that contains all the code in Methods, and all the model weights in the Parameters of the Module. However, each frontend goes through a different pathway for generating those Modules.   
JIT programs通过trace（`torch.jit.trace`）前端或者script（`torch.jit.script`）前端生成。在这两种情况中，这些前端的结果是一个完整的Module，它包含了Methods的所有code，并且模型的权重在Module的Parameters中。然而，每一种前端通过不同的途径生成这些Modules。  

## Tracer ##

[tracer.h](frontend/tracer.h)
[tracer_state.h](frontend/tracer_state.h)

The tracer produces graphs by recording what actual operations are done on tensors.
The entry point from Python into C++ for tracing using `torch.jit.trace` is `_create_method_from_trace`.    
tracer通过记录对张量做的实际操作来产生图。用`torch.jit.trace`从Python到C++的入口点是 `_create_method_from_trace`。

A thread local instance of the TracingState object maintains a mapping between actual data being computed during the trace (e.g. Tensors) stored in IValues, and the abstract `Value*` in the Graph that would compute that value. The functions `void setValueTrace(const IValue&, Value*)` and `Value* getValueTrace(const IValue&)` are used by the tracer to maintain this mapping.    
TracingState对象的线程局部实例维持一个从在trace期间存储在IValues中被计算的的实际数据到在图中将计算value的抽象`Value*`的映射，tracer使用`void setValueTrace(const IValue&, Value*)`和 `Value* getValueTrace(const IValue&)`来维持这种映射。

An initial IValue to Value mapping is setup up between the inputs to the function being traced and symbolic Value inputs to the Graph being constructed. If we are tracing a `torch.nn.Module`, the tracer also adds Parameters and sub-Modules to the Module being constructed that correspond to the Python `torch.nn.Module` being traced.  These values are also added as mapping so that uses of the Parameters in the trace will create uses of the Parameters in the Graph.   
正在被追踪的函数输入和正在被构建的Graph符号Value输入之间建立一个IValue到Value的初始化映射。如果我们追踪一个`torch.nn.Module`，追踪器会添加Parameters和sub-Modules到正在被构建Module，这些模块对应于正在被追踪的Python`torch.nn.Module`。这些值也作为映射添加，以便在追踪中参数的使用将创建Graph中参数的使用。

As the trace runs, individual operators create Nodes in the Graph being traced to record what happens. This code is currently generated per operator in [tools/autograd/gen_variable_type.py](../../../tools/autograd/gen_variable_type.py). It results in code that looks like the following:   
当trace运行时，在被追踪的图中独立的operators创建Nodes以记录发生了什么。这个code由tools/autograd/gen_variable_type.py中的每个operator生成。结果code看起来如下：

```cpp
torch::jit::Node* node = nullptr;
std::shared_ptr<jit::tracer::TracingState> tracer_state;
if (jit::tracer::isTracing()) {
        tracer_state = jit::tracer::getTracingState();
        at::Symbol op_name;
        op_name = jit::Symbol::fromQualString("aten::__ilshift__");
        node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
        jit::tracer::recordSourceLocation(node);
        jit::tracer::addInputs(node, "self", self);
        jit::tracer::addInputs(node, "other", other);
        tracer_state->graph->insertNode(node);

        jit::tracer::setTracingState(nullptr);
}
TypeDefault::__ilshift__(self, other);
if (tracer_state) {
        jit::tracer::setTracingState(std::move(tracer_state));
        jit::tracer::addOutput(node, self);
}
```

The functions `addInputs` and `addOutput` are overloaded to handle the different data types that operators use.
在处理操作符用到的不同数据类型时，函数`addInputs`和`addOutput`被重载

Currently set/getValueTrace only works on Tensors and Futures. Other types are not natively traced. Instead aggregates like tuples or lists are often flattened into tensors at the end of a trace and explicitly constructed from individual tensors at the beginning of this trace.    
目前的set/getValueTrace仅适用于Tensors和Futures，其它类型未被追踪，像tuples或lists在追踪的末尾会被平铺为tensor，并且在追踪开始的时候从独立的张量显示构造。

The tracer has special behavior when tracing calls to other TorchScript functions. This behavior is implemented in the GraphExecutor right before a Graph is about to be run. If tracing is enabled while running the graph, the GraphExecutor will disable tracing, run the graph as normal, and then inline the Graph into the trace. It then hooks up the IValues computed by running the Graph to out Values in the inlined graph.   
当追踪调用TorchScript函数，tracer有特殊的行为，在图运行之前，该行为在GraphExecutor中实现。当运行图时，如果追踪开启，GraphExecutor将禁用追踪，以normal形式运行图，然后内联图到追踪。通过运行图计算IValues以输出内敛图中的Values。

> *When a trace calls a TorchScript function, that function is preserved as is, meaning that control-flow is preserved.* This makes it possible to "fix" tracing issues by writing the subset of the program that cannot be traced in script and having the trace invoke it.   
当一个trace调用一个TorchScript函数，该函数将被原样保留，这意味控制流被保留。这意味着通过写在script中无法被追踪的程序子集并让trace调用它来修复trace问题成为可能。  

The resulting Graph created by tracing is installed as the 'forward' method of the Module being created. A Module is produced regardless of whether the thing being traced was a function or a `torch.nn.Module`. In the function case, the Module produced will simply have a single `forward` function, no Parameters, and no sub-Modules.  
通过追踪创建的结果Graph以正在被创建的Module中的forward方法安装。无论被追踪的是函数还是`torch.nn.Module`，都会产生一个Module。在函数情况中，产生的Module有一个 `forward`函数，没有Parameters，没有sub-Modules。

## Script ##

The script frontend directly converts Python syntax into Modules. Like many compilers this happens in two phases. First, we generate an abstract syntax tree (AST), which is constructed out of Tree objects. The compiler (misnamed, but that is the name of the file) then does semantic analysis on the Tree and lowers it into a Module. We can generate Trees in two ways: (1) using frontend.py, which takes the Python AST and transliterates it into Tree objects, or (2) via the Lexer and Parser which parse python syntax directly.  The Lexer/Parser path may seem redundant but it is crucially important. We need to define builtin functions ([script/builtin_functions.cpp](script/builtin_functions.cpp)) when Python is not linked. We allow users to load TorchScript programs directly from strings without Python ([api/include/torch/jit.h](../../api/include/torch/jit.h)). We also use this Python syntax as the serialization format for TorchScript, since it allows us to make changes to our IR without breaking backward compatibility. Furthermore, the Lexer is reused to implement the FunctionSchema parser, which turns FunctionSchema declarations from strings into FunctionSchema objects.     
script前端直接转化Python syntax到Modules。像大多数编译器一样，该过程有两个阶段。首先，生成一颗抽象语法树（AST），它是由Tree对象构成。编译器（错误的命名，它是文件的名字）在Tree上做语法分析，并将它降低为Module。我们可以使用两种方式生成Tree:(1)用frontend.py接收Python AST并将其转化为Tree对象 (2)直接通过Lexer和Parser解析python语法。Lexer/Parser路径看起来是冗余的，但它是至关重要的。当Python未连接时，我们需要定义内置函数([script/builtin_functions.cpp](script/builtin_functions.cpp))。不用Python([api/include/torch/jit.h](../../api/include/torch/jit.h))允许从字符串加载TorchScript程序。对于TorchScript我们也是用Python语法作为序列化格式，因为它允许在不破坏后端兼容性的同时修改我们的IR。Lexer被用于实现FunctionSchema解析器，它将FunctionSchema声明从字符串转为FunctionSchema对象。


The following sections look into each the stages in the script frontend in detail.   
以下章节介绍script前端的每个阶段

## Tree ##

[frontend/tree.h](frontend/tree.h)

Our frontends produce ASTs in the form of Tree objects. Trees are similar to [s-expressions](https://en.wikipedia.org/wiki/S-expression). Leafs (i.e. Atoms) are always strings. Compound trees have a `kind` (e.g `TK_CONST` or `TK_IDENT` defined in lexer.h) and a list of sub-trees.  For instance, the Tree for `z.sigmoid() - (x + y)` is:  
我们前端以Tree对象的形式生成ASTs。树类似于[s-expressions](https://en.wikipedia.org/wiki/S-expression)，叶子（Atoms）总是字符串。复合树有一个`kind`（在lexer.h中定义的`TK_CONST`或`TK_IDENT`）和sub-trees列表。例如，对于`z.sigmoid() - (x + y)`的树：

```
 (-
        (+
          (variable (ident x))
          (variable (ident y)))
        (apply
          (.
                (variable (ident z))
                (ident sigmoid))
          (list)
          (list))))
```

This is printed in s-expression style with `(kind ...)` representing compound trees and `string_value` representing strings.   
以s-expression风格打印，`(kind ...)`表示复合树，`string_value`表示字符串。

We provide utilities to construct, traverse, and print ASTs without a lot of complicated visitor infrastructure and inheritance.  
提供使用程序来构造、遍历并打印ASTs，而无需复杂的访问者基础设施和继承

Each tree also has a mandatory SourceRange object that describes the range of text that it came from. These will be used for error reporting in the rest of the code.    
每颗树有一个强制性的SourceRange对象，用它来描述来自文本的范围。这些将用于其余代码的错误报告

## Tree Views ##

[frontend/tree_views.h](frontend/tree_views.h)

Trees are easy to construct visualize and traverse, but extracting information from a large compound tree like that of a function definition is unwieldy since it requires numeric indexing. Tree _Views_ are a small layer on top of a tree that make it possible to create and de-structure trees of particular kinds. For example, here is the tree view for the apply node which provides named accessors for its subtrees: the function being called, the inputs, and the attributes (i.e. kwargs):  
Tress很容易构造可视化和遍历，从像函数定义这样的大型复合树提取信息是不实际的，因为它要求数字索引，Tree的_Views_是树顶部的一个小层，可以创建和析够指定类型的树。例如，应用节点的树视图为它的子树提供命名访问器：被调用的函数，输入和属性（即kwargs）：

```cpp
struct Apply : public Expr {
  Expr callee() const {
    return Expr(subtree(0));
  }
  List<Expr> inputs() const {
    return List<Expr>(subtree(1));
  }
  List<Attribute> attributes() const {
    return List<Attribute>(subtree(2));
  ...
};
```

The typical way to traverse a tree is to `switch` on the kind and then construct the appropriate Treeview:  
遍历树的典型方法是在kind上的 `switch`，然后构造适当的Treeview:

```cpp
switch (tree.kind()) {
  case TK_VAR:
        auto var = Var(tree); // construct tree-view
        return environment_stack->getSugaredVar(var.name());
  case '.': {
        auto select = Select(tree); // construct tree-view
        auto sv = emitSugaredExpr(select.value(), 1);
        return sv->attr(select.range(), method, select.selector().name());
  }
  case TK_APPLY: {
        auto apply = Apply(tree); // construct tree-view
        return emitApplyExpr(apply, n_binders);
  } break;

```

## frontend.py ##

[torch/jit/frontend.py](../../jit/frontend.py)

One way we construct Tree objects is directly from Python ASTs. This logic is contained inside frontend.py and is intentionally very minimal.  
构建Tree对象的一种方法是直接从Python的ASTs构建。这个逻辑包含在frontend.py，并且故意非常小。

> *We endeavor to keep most of the JIT code written in C++, because most of the JIT functionality still needs to work without Python installed.*   
我们努力用C++写大部分JIT代码，因为大部分JIT功能仍然需要不安装Python的情况下工作。

So this code simply constructs the Tree, filtering out the AST nodes of Python that we do not support.    
所以这段代码只是简单构造树，过滤掉我们不支持的Python AST节点。

## Lexer ##

[frontend/lexer.h](frontend/lexer.h)

When loading TorchScript code directly from a string, we using a standard Lexer/Parser combo. The Lexer takes an initial string and then exposes a stateful interface for walking the Tokens of the string, providing a standard set of functions:   
当从字符串加载TorchScript code，我们用标准的Lexer/Parser组合。Lexer接受一个初始化的字符串，然后公开一个有状态的接口来遍历字符串的Tokens，提供一组标准函数：

* `next()` advances the lexer, returning the current token    

* `cur()` provides the current token
* `lookahead()` provides the token coming after the current token  
提供当前token之后的token
* `nextIf(int token_kind)` advances the token if it matches token kind.    
如果token匹配token_kind，则next token

Similar to Python, the Lexer handles the white-space sensitive nature of Python blocks. The Tokens `TK_INDENT`, `TK_DEDENT`, and `TK_NEWLINE` are injected into the token stream when code first becomes indented, when it dedents, and at the end of a statement. For instance for this stream:    
类似于Python，Lexer处理Python block空格敏感特性。当代码第一次缩进以及语句的结尾，Token `TK_INDENT`、`TK_DEDENT`、和`TK_NEWLINE`被注入token流，例如对于这个流：

```cpp
if
  .
  .
```

We would get a token stream `TK_IF TK_NEWLINE TK_INDENT . TK_NEWLINE . TK_NEWLINE TK_DEDENT`. Unmatched opening brackets disable the injection of these tokens. The result is that the Parser can simply treat `TK_INDENT`, `TK_DEDENT` and `TK_NEWLINE` like C's `{`, `}`, and `;`.  
我们将获得一个token流`TK_IF TK_NEWLINE TK_INDENT . TK_NEWLINE . TK_NEWLINE TK_DEDENT`。未匹配的左括号将禁用这些tokens的注入。结果是Parser可以像C的 `{`、`}`、和 `;`那样对待 `TK_INDENT`, `TK_DEDENT`和`TK_NEWLINE`。

## Tokens ##

[frontend/lexer.h](frontend/lexer.h)

Tokens are either keywords (`def`), operators (`+`), literals (`3.4`), or identifiers (`foo`). A `token_kind` integer identifies what it is and is the exact same type as the `kind` of a Tree. For single-character Tokens (e.g. `+`), the kind is the same as the character, enable statements like:    
Tokens可以是关键字(`def`)、操作符(`+`)、文字(`3.4`)或者标识符(`foo`)。`token_kind`用整数标识它是什么，并且和Tree的`kind`类型是完全相同的。对于单字符Tokens（例如`+`），种类和字符相同，启用语句如下：

```cpp
if (lexer.nextIf('+')) {
        // handle + ...
}
```

Multi-character token kinds are defined in a list, `TC_FORALL_TOKEN_KINDS`. Tokens also have a `text()` field that records the actual string producing the token and is used by identifiers and literals to construct the actual values (e.g. the numeric value of a floating point literal).   
多字符token种类定义在列表中，`TC_FORALL_TOKEN_KINDS`。Tokens有一个`text()`字段，来记录产生token的实际字符串，并且被identifiers和literals用来构造实际值（例如浮点文字的数值）。

## Parser ##

[frontend/parser.h](frontend/parser.h)

The Parser uses the Lexer to build the AST for function definitions. `parseFunction` is the entrypoint for parsing a single `def ...` and will return a `Def` tree view.  
Parser用Lexer为函数定义构建AST。`parseFunction`是解析单个`def ...`的入口点并且返回`Def`树视图。

The Parser is written as a [top-down precedence parser](https://eli.thegreenplace.net/2010/01/02/top-down-operator-precedence-parsing), or "Pratt" parser.  They are simpler and easier to understand than typical parser generators, while still being flexible enough to parse programming languages. For the most part parsing is done by recursive decent. To resolve operator precedence issues, the function to parse an expression is augmented with a precedent _p_ such that calling the function means _parse an expression whose operators all have precedence higher than p_.     
Parser作为一个自上到下的优先级解释器或者“Partt”解析器。它比传统的解析器生成器更简单、更易理解，同时具有足够的灵活性来解析编程语言。对于大部分情况，解析通过递归decent完成。为了解决操作符优先级问题，解析表达式的函数增加一个优先级p，这样调用一个函数意味着解析所有操作符优先级都高于p的表达式。


## IR Emitter ##

[frontend/ir_emitter.h](frontend/ir_emitter.h)

The file ir_emitter.cpp translates Trees into Modules. The main entrypoint is `defineMethodsInModule` which takes a list of Def Tree Views representing function definitions and adds them as Methods to the module. During the lowering processing _semantic checking_ occurs. The IR emitter checks that all used variables are defined (sometimes called scope checking), and that all values have compatible types (type-checking). During this process it also emits the graph nodes corresponding to each statement in the Tree and generates a FunctionSchema for the whole definition.    
ir_emitter.cpp转换Trees到Modules。主要的入口点是`defineMethodsInModule`，它接受一个表示函数定义的Def Tree Views列表并将它们作为Methods添加到module。在lowering processing期间进行语义检查。IR发射器检查所有使用的变量是否被定义（也称scope检查），还进行类型检查。在这个处理过程中，还发出图节点对应到树中的每个语句，并且对于整个定义生成一个FunctionSchema。

A few helper objects exist in the lowering process.  SugaredValues are special values that represent objects that can appear during compilation but that are not first class values. For instance, in TorchScript methods `self` refers to the module, and `self.weight` refers to a Parameter of the module. Neither are first-class Types and have no corresponding Value in a graph. Resolver objects are std::functions that resolve externally-defined variables to SugaredValues. For instance, the identifier `torch` which contains most of our built-in ops is looked up through Resolver objects which interact with the python state of the program.    
在lowering process中存在一些辅助对象，SugaredValues是一类特殊值，表示在编译期出现但不是第一类值的对象。例如，在TorchScript方法中`self`表示module，`self.weight`表示module的Parameter，两者都不是first-class Types，在图中没有相应的Value。Resolver对象是std::functions，它解析外部定义的变量到SugaredValues。例如，通过与程序的python状态交互的Resolver对象来查找包含大部分内置操作符的 `torch`。

The Environment tracks the mapping between variable names and the SugaredValues they refer to.    
Enviroment追踪变量名和SugaredValues之间的映射。

## SugaredValue ##

[frontend/sugared_value.h](frontend/sugared_value.h)

SugaredValues are how the IR emitter represents non-first class values during Graph creation. These values are things like the Module or a python function call that do not have corresponding Value objects in the Graph. The IR emitter _desugars_ the SugaredValue objects to instructions in the graph based on how they are used.  The SugaredValue class has a number of abstract methods on it such as `attr` or `call`. Consider the expression `self.foo`. For methods, `self` will resolve to a special SugaredValue subclass,  ModuleValue. When the emitter sees `self.foo`, it will then call the ModuleValue function `sv.attr("foo")`, asking the ModuleValue how it should desugar itself when the attribute `"foo"` accessed. If `foo` is a parameter, it would then ensure that the parameter was added to the Method being compiled, and return a `SimpleValue` sugared value that contains the Value object representing the parameter as an input. If `foo` were a sub-Module then it would return another SugaredModule. The method `call` is invoked when the emitter sees the value used as a function call.     
SugaredValues是在图创建过程中IR 发射器表示non-first class values的一种方式。这些values类似Module或python函数，在图中没有相应的Value对象。基于SugaredValue如何使用，IR发射器脱糖SugaredValues到图中的指令。SugaredValues有一些抽象方法像`attr` 或`call`。考虑表达式`self.foo`，对于方法`self`将解析成一个特殊的SugaredValue子类ModuleValue。当发射器看到`self.foo`，它将调用ModuleValue函数`sv.attr("foo")`，当属性`"foo"`被访问时询问ModuleValue如何为自己脱糖。如果`foo`是一个参数，它将确保参数添加到正在被编译的Method中，并返回一个`SimpleValue` sugared值，它包含参数作为输入的Value对象。如果`foo`是一个sub-Module，它将返回Sugaredmodule。当发射器看到value作为一个函数调用，方法`call`被调用。

SugaredValues are also how we interact with Python runtime during the compilation process. For instance, `math.pi` is resolved to 3.1415... by first resolving `math` to a SugaredValue representing accesses to Python modules (PythonModuleValue) whose `attr` function turns python numbers into  `prim::Constant` Nodes in the graph.  
SugaredValues也是在编译过程中与Python运行时交互的方式。例如，`math.pi`被解析成3.1415...，通过第一次解析`math`到一个访问Python modules的SugaredValue（PythonModuleValue），Python modules的`attr`函数转换python numbers到图中的`prim::Constant`节点。

Finally, normal Values are also represented by the SimpleValue SugaredValue in places where it is valid that either a SugaredValue or a normal Value will appear.  
最后，在出现一个SugaredValue或者一个正常Value有效的地方，正常Values由SimpleValue SugaredValue表示。

## Resolver ##

[frontend/ir_emitter.h](frontend/ir_emitter.h)

Any undefined variable during compilation is resolved with a call to an externally-provided Resolver. When called from Python (e.g `torch.jit.script`) this resolver interacts with the Python runtime via pybind11 to resolve symbols like `torch` and `math` to their Python equivalents.    
在编译期间，任意未定义的变量被解析通过调用一个外部提供解析器。当调用来自Python（例如`torch.jit.script`），改解析器和Python运行时通过pybind11交互来解析符号像`torch`和`math`到Python等价的东西上。

*The combination of SugaredValue and Resolver decouples the implementation of the IR emitter from the pybind11 Python bindings that enable its interaction with the Python state.*  
SugaredValue和Resolver的组合将IR 发射器的实现从pybind11 Python绑定中分离，这使得可以与Python状态交互。

This makes it possible to use most of the IR emitter functionality when python is not present.  
当python不存在，可以使用大部分IR发射器的功能。

## Environment ##

[frontend/ir_emitter.h](frontend/ir_emitter.h)

The Environment object tracks the assignment of variable names during compilation. It is local to the IR emitter file. A stack of environments exist, with a new environment being created for sub-blocks introduced by control flow. The Environment keeps two tables, one for values which are not first class in the type system (Sugared values) and a type table for values which are. When first class values are set, we emit a prim::Store, and when they are referenced we emit a prim::Load. Sugared values are not re-assignable. The graph is converted to SSA in the convertToSSA pass.      
Environment对象在编译期间追踪变量名字的分配。对于IR发射器文件，它是本地的。一个environments栈存在，一个新的environment被创建对于通过控制流引入的sub-blocks。Environment维护两个表，一个是values，在类型系统（Sugared values）中不是first class；一个是values类型表。当它们被引用时我们发出一个prim::Load，Sugared values不被重新分配，在convertToSSA路径中，图转化为SSA。


## Conversion To SSA ##

[frontend/convert_to_ssa.cpp](frontend/convert_to_ssa.cpp)

As explained in the * Block * section, the IR is represented in structured control flow composed of ifs & loops. This makes it easier to optimize and lower to other compilers which do not support unstructured control flow. We lower python control flow (break, continue, return) to this simplified form. We do closing over any variables in the environment, so we are able to convert all writes and reads from the environment directly to SSA form.    
正如* Block *章节所述，IR由ifs和loops组成的结构化控制流表示，这使得它很容易优化并lower到其它不支持非结构化控制流的编译器中，我们简化Python控制流（break continue return）到这种简单的形式。我们可以关闭环境中的任何变量，因此我们可以转化环境中的读取和写入到SSA形式。

Conversion to SSA works in multiple parts.  
向SSA转化的工作有以下几个部分：
- First, we add loads and stores to control flow operators (ifs & loops).  
 首先，对于控制流运算符（if & loop）我们添加loads和stores

- Then we erase Break & Continue statements from the graph and replace them with `prim::LoopContinuation`. `prim::LoopContinuation` has the form `LoopContinuation(%loop_continue_condition, %loop_carried_vars)`. Break Statements have the continue condition set to false, and Continue statements inline the loop condition. %loop_carried_vars are the loop carried variables of the inner most loop that contains the Break or Continue statement, are added by inserting prim::Loads calls at the location of the statement.    
然后，我们从graph中删除Break & Continue语句，并且取代它们用`prim::LoopContinuation`，`prim::LoopContinuation`的形式为`LoopContinuation(%loop_continue_condition, %loop_carried_vars)`。Break语句将continue condition设置为false，并且Continue语句内联到loop，%loop_carried_vars为包含Break或者Continue语句的最内层循环的循环携带变量，通过插入prim::Loads调用添加。

- Then we inline the loop condition into the graph loops.  
然后，我们将循环条件内联到graph循环 

- Next we erase loads and stores, removing all Stores and replacing all loads with whatever the in-scope value of the variable name is.    
接下来，删除loads和stores，移出所有的Stores并且用变量名称范围内的值取代所有的loads

- Finally, we remove `prim::LoopContinuation`s and `prim::ReturnStmt`s in the exit_transform pass.  
最后，在exit_transform pass移出所有的`prim::LoopContinuation`和`prim::ReturnStmt`

## Exit Transform ##

[frontend/exit_transform.cpp](frontend/convert_to_ssa.cpp)

This pass takes in a graph where LoopContinuation & ReturnStmts exist in the graph and erases them, correctly setting block outputs. `prim::LoopContinuation(*vals)` means that the values are targeting the most recent loop block. `prim::ReturnStmt(*vals)` means that the values are targeting the most recent Closure or Graph Block.   
pass接受图中的LoopContinuation和ReturnStmts，并删除他们，正确设置输出block。`prim::LoopContinuation(*vals)`意味着这些值针对最近的循环block。`prim::ReturnStmt(*vals)`意味着这些值针对的是最近的Closure或Graph Block。

If a block has an exit node, no further instructions will be executed until the exit target has been reached. If we encounter a node that contains nested blocks that may have hit an exit node, such as an if statement that exits in one block and does not exit in the other, we use a boolean value to indicate if the exit has been hit or not. Then, we conditionalize further execution.    
如果一个block有一个退出节点，在到达退出目标之前，指令将不被执行。如果我们遇到了一个节点包含嵌套块，该嵌套块可能命中一个退出节点，例如，在一个block退出，在其它block中没有退出，我们用一个bool值來指示是否退出被命中。然后，进一步执行条件化。

Python example:

```python
while i < 5:
  if i == 3:
    i += 1
    continue
  i += 2
```

-> transforms to

```python
continue_loop = i < 5
while continue_loop:
  if i == 3:
    i = i + 1
    continue_loop = i < 5
    did_exit = True
  if did_exit:
    pass
  else:
    i = i + 2
    continue_loop = i < 5
```

The pass also keeps track of nodes or blocks that will always throw Exceptions so that we do not unnecessarily conditionalize execution. In the following example, we can treat the if statement as always Returning and remove the `print` statement.    
pass追踪总是抛出异常的nodes或blocks，这样我们可以不用不必要的条件执行。在如下的例子中，我们可以将if语句视为Returning和删除`print`的语句。

```python
if i < 0:
  raise Exception("Negative input")
else:
  return math.sqrt(i)
print(i)  # unreachable code
```

In the above example, the if statement will have one output, with the value on the false branch being `math.sqrt(i)`. In the true branch, insert and use
`prim::Uninitialized`. These are values inserted by the compiler when it can prove the value will never be used. It can be introduced by exceptions, breaks, continues, and returns.    
在上面的例子中，if语句有一个输出，在false分支上的值变为`math.sqrt(i)`。在true分支上，插入并使用`prim::Uninitialized`。编译器证明这些值永远不会使用，这些值被插入。它可以通过exceptions、breaks
continues和returns引入。

We initially considered doing the Transform pass before Loads and Stores were removed from the graph. However, this breaks when a loop carried variable
is captured in a break or continue and then is refined in the rest of the loop body. In the below example, at the point of the `continue`, `x` has type `Optional[int]` but is refined to `int` after the continue statement.  
我们最初考虑做Transform pass之前，移除图中的Loads和Stores。然而，当一个循环携带变量被捕获并在剩余的循环体内被细化时，将发生中断。在如下的例子中，在`continue`处，`x`的类型为`Optional[int]` ，但是在之后的continue语句中，它被细化为`int`。

```python
...
if cond:
  if i < 3:
      x = torch.jit.annotate(Optional[int], None)
      continue
  x = 1
else:
  x = 2
print(x)
```
If we were to rearrange the graph before loads & stores were removed:  
如果在loads和stores被移除前，重排图：

```python
if cond:
  if i < 3:
    x = torch.jit.annotate(Optional[int], None)
    did_continue = True
    continue
  else:
    did_continue = False
  if not did_continue:
    x = 1
else:
  x = 2
if not did_continue:
  print(x)
```
The type of `x` at the print statement would be `Optional[int]`, which breaks its original type.  
print语句中的类型`x`将是`Optional[int]`，这会破坏它的原始类型。

## Python-Compiler Interaction ##  

[python/script_init.cpp](python/script_init.cpp)

A set of special SugaredValues are used to translate between objects in the Python environment and Values in the Graph during the compilation process. The entry-point for this behavior is `toSugaredValue(py::object obj, ...)` which takes a pybind11 Python value and figures out how to turn it into an appropriate SugaredValue. Values exist to represent Python functions, Python modules, and ScriptModule objects.    
在编译过程中，一组特殊的SugaredValues用于在Python环境中的对象和图中的Values之间进行转换。此行为的入口点是`toSugaredValue(py::object obj, ...)`，它接受一个pybind11 Python值并转化它为一个合适的SugaredValue。Values的存在是为了表示Python functions、Python modules和and ScriptModule objects。

# Executing Programs #

TorchScript is executed using an interpreter attached to a JIT-optimizer and compiler. The entry-point for execution is the GraphExecutor object that is created on demand inside a Method when the method is first called. This section first goes over the semantics of graphs, i.e. what does it mean to execute a graph? And then details how the implementation works.  
TorchScript使用附加到一个JIT-optimizer和编译器的解释器执行。执行的入口点是GraphExecutor对象，它是在首次调用method的时候，在Method内部按需创建。本节将讨论图的语义，也就是执行图意味着什么？然后讨论实现的细节。

## Evaluation Semantics ##

TorchScript programs implement a very small subset of Python of that is necessary to run models.  
TorchScript programs实现了一个运行model的非常小的python子集

TorchScript includes immutable value types:  
TorchScript包括不可变的值类型：
* `int`
* `float`
* `Tuple[T0, T1, ...]`

As well as mutable reference types:  
以及可变的引用类型：
* `Tensor`
* `List[T]`
* `Dict[K, V]`

A value of a reference type points to an underlying memory location where the data for the reference type is stored, and variable assignment for a reference type can cause multiple values to point to the same underlying data. This is similar to Python's class model.    
引用类型的value指向引用类型存储数据的底层内存位置，并且对一个引用类型变量赋值，可造成多个values指向相同的data。这类似于python的类型模型。

It is important to remember that TorchScript uses these semantics for Tensors so not all computation on Tensor is pure. Individual Tensors may be *views* of the same underlying data. Views are established by special view creating operations, such as indexing into a tensor:  
重要的是要记住，TorchScript对张量使用这些语义，因此并非所有张量上的计算都是纯的，独立的张量可能是相同基础数据的视图。Views被特殊的视图创建操作创建，像tensor的索引：  
```python
t = torch.rand(3, 4)
t2 =  t[0] # view of one slice of t
```

Some builtin operators also mutably write to the underlying tensor. In the standard library these operators are always named with a trailing underscore, or take a named `out` tensor where the result is written:  
一些内置操作符也可变的写入底层张量，在标准库中这些操作符通常用一个尾部下划线命名，或者在写入结果的地方接受一个名为out的tensor：

```python
t2.relu_() # inplace relu operator, note t is modified as well!
torch.add(t, t, out=t) # update t, without using temporary memory if possible
```

The combination of reference semantics and mutable operators can be more difficult to optimize, but it gives program writers powerful control of the memory usage of their programs. For instance, DenseNets use a concat operation instead of the addition found in a ResNet. Rather than compute a concat of existing tensors, many implementations use Tensor indexing and `out` keywords to avoid allocating addition memory for the activations. Ideally a compiler would always be able to do these optimizations, but in practice new ideas are tried all the time that exist outside what compiler writers expect and these manual operators allow users to get decent behavior before the compilers catch up.    
引用语义和mutable操作符的组合对优化是困难的，但是它给程序作者提供了更强的内存使用的能力。例如，DenseNets使用concat而不是ResNet中addition。大多数实现用Tensor索引和`out`关键字来避免activations额外的内存分配，而不是计算现有张量的concat。理想情况下，编译器能够总是做这些优化，但是在实践中，总是会尝试新的想法，这些想法超出了编译器作者的预期，这些手工算子在编译器赶上之前可以获得体面的行为。

In addition to being mutable, tensors also have a set of dynamically determined properties (i.e. properties that can vary from run to run) this includes:  
除了可变之外，tensor也有一些动态确定的属性（例如，可以随着运行变化的属性），这包括：

* dtype - their data type int, float, double, etc.  
  张量的数据类型
* device - where the tensor lives, e.g. the cpu, or cuda gpu 0  
  张量所在的设备
* rank - the number of dimensions that the tensor has  
  张量的维度数量
* size - the precise size of the tensor  
  张量的大小
* requires_grad - whether the tensor is recording its gradient with autograd  
  张量是否用autograd记录梯度

Changes in these properties change how operators on tensor will evaluate and would make certain optimization invalid. For instance, if we have fuser capable of generating new cuda kernels but not cpu kernels, it is only valid to fuse operations where the inputs are known to run only on CUDA devices. The GraphExecutor's job is to still enable optimization even when certains combinations of properties prevent optimizations from occurring.  
属性的变化会改变tensor上算子的评估并使某些优化失效。例如，如果我们的fuser能够生成新的cuda内核而没有生成cpu内核，在已知输入运行在CUDA设备上，fuse操作才是有效的。GraphExecutor在某些属性组合阻止优化的时候仍然开启优化。  

Nodes in a graph are executed *serially* in the order they appear in a block. Nodes may be reordered either during optimization or by the interpreter itself if it can be proven that
it is not distinguishable from the original serial execution order. These semantics are necessary since the combination of mutable tensors and potential alias between tensors makes it unsafe to perform arbitrary reordering otherwise. However, the AliasInfo object can accurately track how alias propagate through builtin operators so optimization passes can query when certain reorders or optimizations are safe.    
图中的Node按照出现在block中的顺序执行。如果不能和原来的执行顺序进行区分，nodes可能在优化期间或者通过解释器重新排序。这些语义是重要的，因为可变tensor的组合和tensor之间潜在的别名使得执行任意重新排序变得不安全。然而，AliasInfo对象可以追踪别名如何在内置操作符中传播，这样优化pass可以查询重排或者优化何时是安全的。

We also provide user-accessible parallel execution through the `fork` and `wait` primitives. The `fork` primitive begins execution of `fn` in parallel with the current thread of execution, immediately returning a Future object that will hold the result of the forked function. The `wait` method of the future then causes the invoking task to wait for the value being computed on the forked task.  
我们也提供用户可访问的并行执行通过`fork`和`wait`原语。`fork` 原语与目前执行的线程一起并行执行`fn`，返回一个保存fork函数结果Future对象，future的`wait`方法将导致调用任务等待forked任务计算的值。

```python
def fn(arg0, arg1, ...):
  ...
  return v

fut = torch.jit._fork(fn, arg0, arg1, ...)
...
v = torch.jit._wait(fut)

```

Currently, the user is responsible for avoiding racing immutable operations between tasks. We encourage users to not write to tensors visible from other threads, and may enforce this more strictly in the future.  
目前，用户负责避免任务之间竞争性的不可变操作。我们鼓励用户不要写入来自其它线程可见的张量，在未来可能更加严格的执行该规则。

Optimization passes that wish to exploit multi-threaded execution may automatically convert serial Blocks into parallel execution by inserting extra fork and wait events. This design enables our users to manually specify parallelism while also allowing optimization passes to exploit it when safe and profitable.  
优化pass希望利用多线程执行，通过插入fork和wait事件，可能自动转化串行Blocks到并行执行。这种设计使我们用户手动指定并行性同时当安全和有利可图时允许优化pass利用它。

## IValue ##

[ivalue.h](../../include/ATen/core/ivalue.h)

All evaluation involves computation using IValues, a 16-byte tagged union that can hold the concrete representation of any type in TorchScript. TorchScript is statically typed, so it would be possible to operate on unboxed primitive types, but the interface between interpreter, builtin-ops and user functions would be significantly more complicated. A single tagged union keeps these interfaces simple and since more objects are Tensors anyway, the overhead of storing a tag is small compared to the data stored in the tensors.  
所有的evalution都涉及用IValue计算，一个16字节的标签union可以存储TorchScript中任意类型的正确表示。TorchScript是静态类型，它可以在unboxed原始类型上进行操作，但是解释器、内置操作符和用户函数之间的接口明显是更复杂的。单一的标记union使这些接口简单并且由于很多对象都是张量，和存储数据的张量相比，存储标签的开销是小很多的。

IValue contains methods to check the type `isTensor` and to convert to particular to type `toTensor`. We do not publicly expose the type tag and force clients to use the `isX` methods. This enables us to change the underlying implementation of IValue later, e.g. to use an 8-byte value with NaN-boxing. Most operators work on a specific static type, so dynamic dispatch on the tag is not frequently required.  
IValue包含检查类型`isTensor`方法和转到特定类型`toTensor`的方法。我们没有公开暴露类型标签并且强迫用户使用`isX`方法，这使得我们以后可以更改IValue的底层实现，例如，用一个NaN-boxing的8字节value。大多数操作符在特定的静态类型上工作，因此不需要经常在tag上动态调度。

## Operation ##

All builtin operators are represented using a stack machine concept. An operator pops its arguments off the top of the stack and pushes its result to the stack:  
所有的内置操作符用一个栈machine概念来表示，一个操作符从栈顶弹出它的参数并将结果push到stack。

```cpp
using Stack = std::vector<IValue>;
using Operation = std::function<int(Stack&)>;

// schema: example_add(Tensor a, Tensor b) -> Tensor
int example_add(Stack& stack) {
    Tensor a, b;
    // stack before: ? ? ? a b <- back
    pop(stack, a, b); //Templated helper function
                      // that pops a, b and converts them to tensor
    push(stack, a + b);
    // stack after:
    // ? ? ? c <- back
    return 0; // goto the next instruction
}
```

Most operations, apart from some vararg primitive operators like prim::Unpack, have an associated FunctionSchema that describes how many inputs will be popped and how many will be pushed.  
大多数操作符，除了像prim::Unpack这一类的可变参数原始操作符之外，都有一个关联的FunctionSchema，它描述了多少输入将被pop，多少将被push。

The stack concept makes it easy to define operators with variable numbers of inputs and outputs without the need to allocate vectors of inputs and outputs for each individual operator.  
stack概念使得定义可变数量的输入和输出操作符变得容易，不需要为每一个单独的操作符分配输入和输出向量。

In practice, the interpreter will allocate one Stack, and it will eventually reach a sufficient size such that no more stack-related memory allocations will occur.  
在实践中，解析器将分配一个栈，最终达到足够的大小，这样与栈相关的内存分配将不再发生。

Operations also return a jump offset relative to the address of the next operator in the program to for dynamic control flow. Except for special Operations in the interpreter that handle control-flow all Operations should return 0 here. It is a bit weird to force all Operations to return 0, but it avoids having to have another level of indirection to wrap void functions in something that returns 0.  
在程序中，操作符返回一个对于下一个操作符的地址偏移，以用于动态控制流。除了在解释器中处理控制流的特殊操作符外，所有的操作符必须返回0,这有一点奇怪强制所有操作符返回0,它避免了有其它间接级别在返回0的东西中包裹void函数。

## Operator ##

[runtime/operator.h](runtime/operator.h)

The Operator object represents a single registered operator in the system. It combines a FunctionSchema that describes how an Operation executes with a method to lookup the corresponding Operation given the Node representing the operator in a Graph.  Most Operators are defined by providing a FunctionSchema and an Operation function. However, primitives like prim::Unpack require knowledge of their Node to know how to operate (e.g. how many elements to unpack). These Operators have a function that takes a `Node*` and returns an operation.   
Operator对象表示在系统中注册的一个单一操作符，它结合了一个FunctionSchema（描述操作如何执行）和一个method，以便给出一个在图中代表操作符的Node来查找相应的Operation。大多数操作符通过提供一个FunctionSchema和一个Operation函数来定义，然而像prim::Unpack这样的原语，只有知道了它们的Node之后才知道如何操作（例如，多少元素要解包）。这些Operator有一个函数接受`Node*`并返回一个operation。

## Interpreter ##

[runtime/interpreter.cpp](runtime/interpreter.cpp)

The interpreter is responsible for the straightforward execution of Graphs without any optimization. It is composed of two objects: Code and InterpreterState. Code is a linearized representation of the Graph into simple stack-machine Instructions. Code is shared among all the executions of the Graph and will include caches for certain operations like the generated CUDA code of FusionGroups.  
解释器负责直接执行图，而不进行任何优化，它由两个对象组成：Code和InterpreterState。Code是Graph的线性化表示，是简单的栈机器指令。在图的执行过程中Code是被共享的并且包括某些操作的缓存，例如FusionGroups生成的CUDA代码。

The InterpreterState is unique to each execution of the Graph. It holds a list registers with the intermediate IValues used in the execution, the Stack being used by each Operation, and the program counter tracking the position in the instructions. The information represents the complete state of the interpreter. `wait` instructions can cause the interpreter to suspend, and the InterpreterState is used to resume execution where the `wait` occurred, potentially on a different thread.   
图每次执行InterpreterState都是唯一的，它保存了一个列表注册器，包含执行过程中用到的中间IValues，每个Operation用到的Stack，在指令中追踪位置的程序计数器，InterpreterState表示解释器的完整状态。`wait`可以造成解释器挂起，InterpreterState会在`wait`的地方恢复执行，可能是在一个不同的线程中。

Instructions in the interpreter have three parts: a list of registers from which to gather IValues onto the stack before the instruction, the Operation to run, and a list of registers in which to store the results of the Operation. Alternatively, we could have used individual instructions to load/store values from the stack to registers, but this design was easier to implement, requires fewer instructions since each instruction does more things, and has not yet been a performance bottleneck. Each Operation returns a potential relative jump to compute the next program counter.  
指令在解释器中有三部分：  
* 寄存器列表，用于在指令之前收集IValues到stack
* 要运行的Operation
* 寄存器列表，存储Operation的结果    

另外，我们可以用单独的指令从栈到寄存器加载/存储值，但是这种用设计更容易实现，需要更少的指令，因为每个指令做更多的事，并且还没有成为性能瓶颈。每个Operation返回一个潜在的相对跳转来计算下一个程序计数器。

Unlike typical interpreters, we not attempt to do careful register allocation. Since Tensors are reference types, saving registers would only save a few hundred bytes of space in typical applications by cutting down on the number of places a reference could be saved. The data in single a Tensor is likely significantly bigger than that, so we forgo register allocation to make debugging easier.  
与典型的解释起不同，我们没有仔细地做寄存器分配。因为Tensors是引用类型，在典型的应用中，通过减少引用的保存位置，保存寄存器只会节省几百个字节的空间。单个张量中的数据可能比这大的多，所以我们放弃了寄存器分配来使调试更容易。

However, we do need to ensure that values are destructed immediately after their last use. Because Torch reference counts Tensors, they will be deallocated immediately when their last reference is gone. To ensure we use a minimum amount of memory we want to ensure that the interpreter releases the reference as soon as it is no longer used. To do this, each Instruction also has a set of flags which indicate the inputs to the operation which will no longer be used after the operation. For these inputs, the IValue is moved rather than copied from the register file, ensuring the reference will go dead as soon as the Operation no longer needs it.  extra instructions may be inserted into the program to explicitly drop references for values whose last use depends on the control flow of the program.    
然而，我们必须确保值在最后一次使用后立即被析够，因为Torch对Tensor的引用计数，所以Tensor的最后一次引用消失后，它将立即被析构。我了确保我们使用最少的内存，我们需要确保引用不再使用的时候，解释器立即释放它。为了做到这些，每个Instruction有一组标志来指示操作的输入，这些输入在operation后将不再使用。对于这些输入，IValue被move而不是从寄存器文件中被copy，确保Operation不需要它时候，引用就会失效。可以在程序中插入额外的指令来显式删除值的引用，这些值的最后使用依赖程序控制流。

```
graph(%x : Tensor,
      %hx : Tensor,
      %cx : Tensor,
      %w_ih : Tensor,
      %w_hh : Tensor,
      %b_ih : Tensor,
      %b_hh : Tensor):
  %7 : int = prim::Constant[value=4]()
  %8 : int = prim::Constant[value=1]()
  %9 : Tensor = aten::t(%w_ih)
  %10 : Tensor = aten::mm(%x, %9)
  %11 : Tensor = aten::t(%w_hh)
  %12 : Tensor = aten::mm(%hx, %11)
  %13 : Tensor = aten::add(%10, %12, %8)
  %14 : Tensor = aten::add(%13, %b_ih, %8)
  %gates : Tensor = aten::add(%14, %b_hh, %8)
  %16 : Tensor[] = aten::chunk(%gates, %7, %8)
  %ingate.1 : Tensor, %forgetgate.1 : Tensor, %cellgate.1 : Tensor, %outgate.1 : Tensor = prim::ListUnpack(%16)
  %ingate : Tensor = aten::sigmoid(%ingate.1)
  %forgetgate : Tensor = aten::sigmoid(%forgetgate.1)
  %cellgate : Tensor = aten::tanh(%cellgate.1)
  %outgate : Tensor = aten::sigmoid(%outgate.1)
  %25 : Tensor = aten::mul(%forgetgate, %cx)
  %26 : Tensor = aten::mul(%ingate, %cellgate)
  %cy : Tensor = aten::add(%25, %26, %8)
  %28 : Tensor = aten::tanh(%cy)
  %hy : Tensor = aten::mul(%outgate, %28)
  %30 : (Tensor, Tensor) = prim::TupleConstruct(%hy, %cy)
  return (%30)
```

```
0, 1, 2, 3, 4, 5, 6 = Load
7 = Constant
8 = t move(3)
9 = mm move(0), move(8)
10 = t move(4)
11 = mm move(1), move(10)
12 = add move(9), move(11), 7
13 = add move(12), move(5), 7
14 = add move(13), move(6), 7
15, 16, 17, 18 = ConstantChunk move(14)
19 = sigmoid move(15)
20 = sigmoid move(16)
21 = tanh move(17)
22 = sigmoid move(18)
23 = mul move(20), move(2)
24 = mul move(19), move(21)
25 = add move(23), move(24), move(7)
26 = tanh 25
27 = mul move(22), move(26)
28 = TupleConstruct move(27), move(25)
 = Store move(28)
```

## Graph Executor ##

[runtime/graph_executor.cpp](runtime/graph_executor.cpp)

All program execution starts with a graph executor. Its responsible for running optimizations (potentially involving the JIT-compilation of fused kernel code), and then handing the Graph or subcomponents of it off to an interpreter to actually run.  
所有程序的执行都是从graph executor开始的，它负责优化（可能涉及融合内核代码的JIT编译），然后将图或者其组件交给解释器执行。

In this section, we use a running example program that computs one step of a LSTM to show how the graph is transformed:  
在本节中，我们用一个LSTM的例子展示图如何被转换的：

This section will use an example this LSTM program:  
本节使用这个LSTM程序：

```python
@torch.jit.script
def LSTMCellS(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
    gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy
```

After going through the the frontend, we get start with this unoptimized graph:  
通过前端，我们获得一个未优化的图：

```
graph(%x : Tensor,
      %hx : Tensor,
      %cx : Tensor,
      %w_ih : Tensor,
      %w_hh : Tensor,
      %b_ih : Tensor,
      %b_hh : Tensor):
  %7 : int = prim::Constant[value=4]()
  %8 : int = prim::Constant[value=1]()
  %9 : Tensor = aten::t(%w_ih)
  %10 : Tensor = aten::mm(%x, %9)
  %11 : Tensor = aten::t(%w_hh)
  %12 : Tensor = aten::mm(%hx, %11)
  %13 : Tensor = aten::add(%10, %12, %8)
  %14 : Tensor = aten::add(%13, %b_ih, %8)
  %gates : Tensor = aten::add(%14, %b_hh, %8)
  %16 : Tensor[] = aten::chunk(%gates, %7, %8)
  %ingate.1 : Tensor, %forgetgate.1 : Tensor, %cellgate.1 : Tensor, %outgate.1 : Tensor = prim::ListUnpack(%16)
  %ingate : Tensor = aten::sigmoid(%ingate.1)
  %forgetgate : Tensor = aten::sigmoid(%forgetgate.1)
  %cellgate : Tensor = aten::tanh(%cellgate.1)
  %outgate : Tensor = aten::sigmoid(%outgate.1)
  %25 : Tensor = aten::mul(%forgetgate, %cx)
  %26 : Tensor = aten::mul(%ingate, %cellgate)
  %cy : Tensor = aten::add(%25, %26, %8)
  %28 : Tensor = aten::tanh(%cy)
  %hy : Tensor = aten::mul(%outgate, %28)
  %30 : (Tensor, Tensor) = prim::TupleConstruct(%hy, %cy)
  return (%30)
```

Execution starts in `GraphExecutor::run`, which takes a Stack of inputs.  
在`GraphExecutor::run`开始执行，它接受一个输入栈。

*Specialization* The executor *specializes* the Graph for the particular set of inputs. Specialization is handled by the `ArgumentSpec` object which extracts a "signature" composed of all the properties being specialized. We only specialize to the properties of Tensors. The ArgumentSpec only records properties for Tensors that either appear directly in the inputs to the graph or inside Tuples that are inputs to the Graph. The properties recorded are currently:  
*Specialization*执行器对于特定的输入集具体化图。具体化由`ArgumentSpec`对象处理，它提取被具体化属性组成的"signature"，我们只具体化Tensors属性。ArgumentSpec只记录直接出现在图输入中Tensor的属性，或者作为图输入Tuple中Tensor的属性，目前被记录的属性有：

* dtype
* rank, but not size
* requires_grad
* device type (cpu, cuda)
* defined - whether the Tensor exists or is a placeholder

The ArgumentSpec object is used as a key into a cache that holds pre-optimized Code objects (held in an ExecutionPlan object). On a cache hit, an InterpreterState is created and the Code in the cache is run.  
ArgumentSpec对象作为保存预优化Code对象（保存在ExecutionPlan对象）的缓冲区中的一个键。缓存命中时，一个InterpreterState被创建，并且cache中的Code被运行。

*Pre-derivative Optimization* On a code cache miss, we generate a new optimized Graph on the fly (`compileSpec`). It starts by creating a copy of the initial Graph and setting the input types to the specialized Tensor types observed in this specialization. TensorType inputs to the Graph will get refined with types that know the device, number of dimensions, and requires grad state.  
预衍生优化 在code缓存未命中时，我们动态生成一个新的优化图（`compileSpec`），它首先创建一个初始化的图的拷贝，并将输入类型设置为在specialization中观察到的具体化Tensor类型。图的输入TensorType将使用已知设备类型、维度数目和grad状态进行细化。

```
# post specialization, inputs are now specialized types
# 后具体化，输入现在是具体化的类型
graph(%x : Float(*, *),
      %hx : Float(*, *),
      %cx : Float(*, *),
      %w_ih : Float(*, *),
      %w_hh : Float(*, *),
      %b_ih : Float(*),
      %b_hh : Float(*)):
  %7 : int = prim::Constant[value=4]()
  %8 : int = prim::Constant[value=1]()
  %9 : Tensor = aten::t(%w_ih)
  %10 : Tensor = aten::mm(%x, %9)
  %11 : Tensor = aten::t(%w_hh)
  %12 : Tensor = aten::mm(%hx, %11)
  %13 : Tensor = aten::add(%10, %12, %8)
  %14 : Tensor = aten::add(%13, %b_ih, %8)
  %gates : Tensor = aten::add(%14, %b_hh, %8)
  %16 : Tensor[] = aten::chunk(%gates, %7, %8)
  %ingate.1 : Tensor, %forgetgate.1 : Tensor, %cellgate.1 : Tensor, %outgate.1 : Tensor = prim::ListUnpack(%16)
  %ingate : Tensor = aten::sigmoid(%ingate.1)
  %forgetgate : Tensor = aten::sigmoid(%forgetgate.1)
  %cellgate : Tensor = aten::tanh(%cellgate.1)
  %outgate : Tensor = aten::sigmoid(%outgate.1)
  %25 : Tensor = aten::mul(%forgetgate, %cx)
  %26 : Tensor = aten::mul(%ingate, %cellgate)
  %cy : Tensor = aten::add(%25, %26, %8)
  %28 : Tensor = aten::tanh(%cy)
  %hy : Tensor = aten::mul(%outgate, %28)
  %30 : (Tensor, Tensor) = prim::TupleConstruct(%hy, %cy)
  return (%30)
```

It then runs "required passes", which are graph transformations necessary to generate legal graphs for the interpreter. (Some passes such as differentiation will introduce Nodes that are not defined by operators and require passes to clean up. The combination of `specializeUndef` and `LowerGradOf` clean up these operations.) These passes also remove broadcasting "expand" nodes that get implicitly inserted by the tracer but are not valid for all sizes.  
然后运行“必须的passes”，这是解释器生成合法图必要的图形转换。（像微分一类的passes将引入不是由操作符定义的节点，并且需要passes来清理， `specializeUndef`和`LowerGradOf`的组合来清理这些操作）这些passes也移除由追踪器隐式差入的广播扩展节点，但是对所有的sizes并不都是有效的。

It then runs inference passes to calculate properties of the graph given this particular specialization:  
然后运行inference passes来计算给定图的属性

* It propagates constants, pre-computing as much as possible  
传播常数，尽可能的预先计算
* It propagates the input ranks, dtypes, devices, and requires_grad information to the rest of the graph where possible.  
传播输入ranks、dtypes、devices和requires_grad信息到图的其余部分


```
graph(%x : Float(*, *),
      %hx : Float(*, *),
      %cx : Float(*, *),
      %w_ih : Float(*, *),
      %w_hh : Float(*, *),
      %b_ih : Float(*),
      %b_hh : Float(*)):
  %8 : int = prim::Constant[value=1]()
  %9 : Float(*, *) = aten::t(%w_ih)
  %10 : Float(*, *) = aten::mm(%x, %9)
  %11 : Float(*, *) = aten::t(%w_hh)
  %12 : Float(*, *) = aten::mm(%hx, %11)
  %13 : Float(*, *) = aten::add(%10, %12, %8)
  %14 : Float(*, *) = aten::add(%13, %b_ih, %8)
  %gates : Float(*, *) = aten::add(%14, %b_hh, %8)
  %31 : Float(*, *), %32 : Float(*, *), %33 : Float(*, *), %34 : Float(*, *) = prim::ConstantChunk[chunks=4, dim=1](%gates)
  %ingate : Float(*, *) = aten::sigmoid(%31)
  %forgetgate : Float(*, *) = aten::sigmoid(%32)
  %cellgate : Float(*, *) = aten::tanh(%33)
  %outgate : Float(*, *) = aten::sigmoid(%34)
  %25 : Float(*, *) = aten::mul(%forgetgate, %cx)
  %26 : Float(*, *) = aten::mul(%ingate, %cellgate)
  %cy : Float(*, *) = aten::add(%25, %26, %8)
  %28 : Float(*, *) = aten::tanh(%cy)
  %hy : Float(*, *) = aten::mul(%outgate, %28)
  %30 : (Float(*, *), Float(*, *)) = prim::TupleConstruct(%hy, %cy)
  return (%30)
```

It then runs a number of *derivative preserving* optimization passes. If a computation the graph `requires_grad` and it is valid to compute its derivative, then these passes are only allow to replace that computation with another computation that is also differentiable. In other words, these passes cannot break the ability for autograd to work correctly. Algebraic rewrites and peephole optimizations are generally derivative preserving but something that generates code, like pointwise fusion, is not. Currently the passes:  
然后运行一些导数保留优化passes。如果图中的某个计算`requires_grad`，并且计算它的导数是有效的，那么这些passes只允许用另一个同样可微的计算来取代该计算。换句话说，这些passes不能破坏autograd正确工作的能力。代数重写和peephole优化通常是导数保留，但是生成代码的东西，像逐点融合这不是，当前的passes:

* Eliminating dead code  
消除死代码
* Eliminating common subexpressions  
消除常见的子表达式
* Pooling redundant constants into single values  
将冗余常量合并为单个值
* Peephole optimizations, including some algebraic rewrites into simpler operations  
peephole优化，包括将一些代数重写为更简单的操作
* Unrolling small loops.  
展开小循环
* Batching matrix multiplications that result from unrolling loops.    
由展开循环批处理矩阵乘法

```
graph(%x : Float(*, *),
      %hx : Float(*, *),
      %cx : Float(*, *),
      %w_ih : Float(*, *),
      %w_hh : Float(*, *),
      %b_ih : Float(*),
      %b_hh : Float(*)):
  %8 : int = prim::Constant[value=1]()
  %9 : Float(*, *) = aten::t(%w_ih)
  %10 : Float(*, *) = aten::mm(%x, %9)
  %11 : Float(*, *) = aten::t(%w_hh)
  %12 : Float(*, *) = aten::mm(%hx, %11)
  %13 : Float(*, *) = aten::add(%10, %12, %8)
  %14 : Float(*, *) = aten::add(%13, %b_ih, %8)
  %gates : Float(*, *) = aten::add(%14, %b_hh, %8)
  %31 : Float(*, *), %32 : Float(*, *), %33 : Float(*, *), %34 : Float(*, *) = prim::ConstantChunk[chunks=4, dim=1](%gates)
  %ingate : Float(*, *) = aten::sigmoid(%31)
  %forgetgate : Float(*, *) = aten::sigmoid(%32)
  %cellgate : Float(*, *) = aten::tanh(%33)
  %outgate : Float(*, *) = aten::sigmoid(%34)
  %25 : Float(*, *) = aten::mul(%forgetgate, %cx)
  %26 : Float(*, *) = aten::mul(%ingate, %cellgate)
  %cy : Float(*, *) = aten::add(%25, %26, %8)
  %28 : Float(*, *) = aten::tanh(%cy)
  %hy : Float(*, *) = aten::mul(%outgate, %28)
  %30 : (Float(*, *), Float(*, *)) = prim::TupleConstruct(%hy, %cy)
  return (%30)
```

*Post-derivative optimization* The next optimization depends on whether any part of the graph actual requires a gradient to be calculated, which is determined by `needsGradient`. In the case where no gradients are required (i.e. for inference graphs), then we can directly apply optimizations that generate graphs that may not have valid gradients defined. For now this is the `FuseGraph` pass, which looks for adjacent point-wise operations along with reviewing operations such as `split` and `concat`, and creates `prim::FusionGroup` Nodes in the graph to replace these operations. The Operator registered to execute `prim:FusionGroup` nodes will generate a new CUDA kernel for each unique Node, which replaces the original separate execution.  
后导数优化，下一步优化取决于图的任意部分是否需要计算梯度，由`needsGradient`决定是否计算梯度，在这种不需要梯度的情况下（例如推理图），我们可以直接使用优化生成图，不需要有效的梯度定义。现在使用`FuseGraph`pass，它沿着review operation查找相邻的逐点操作，像`split`和`concat`，并在图中创建`prim::FusionGroup`节点来取代这些操作。为执行`prim:FusionGroup`节点注册的Operator将为每一个单独的节点生成一个新的CUDA内核，来取代原来的单独执行。

Note the two phases for compilation of fusion groups: First, the `FuseGraph` pass splits the Graph into fusible sub-Graphs and returns the resulting Graph to the graph executor. Second, when the Graph is turned into Code, the Operation for the FusionGroup node will be looked up and a new CUDA kernel generated for the body. Other compilers should work in a similar way by first introducing a new operator into the Graph where the compiled code should run, and then registering an Operator that implements that Node which performs the actual compilation.  
注意fusion gropus编译的两个阶段：首先，`FuseGraph` pass将Graph拆分成可融合的子图，并将得到的图返回给图执行器。其次，当Graph转化成Code，会查找FusionGroup node的Operation并且为body生成一个新的CUDA kernel。其它的编译器也是以类似的方式工作，首先在应该运行编译code的图中引入一个新的操作符，然后注册一个实现该节点的Operator，执行实际的编译工作。

In the case where no gradients are required, the optimization process is finished, a Code object is constructed from the Graph, it is added to the code cache, and then an InterpreterState is constructed and run.  
在不需要梯度的情况下，优化过程结束，从Graph中构造一个Code对象，并添加到code cache，然后一个InterpreterState被构造并运行。

```
graph(%x : Float(*, *),
      %hx : Float(*, *),
      %cx : Float(*, *),
      %w_ih : Float(*, *),
      %w_hh : Float(*, *),
      %b_ih : Float(*),
      %b_hh : Float(*)):
  %9 : Float(*, *) = aten::t(%w_ih)
  %10 : Float(*, *) = aten::mm(%x, %9)
  %11 : Float(*, *) = aten::t(%w_hh)
  %12 : Float(*, *) = aten::mm(%hx, %11)
  %77 : Tensor[] = prim::ListConstruct(%b_hh, %b_ih, %10, %12)
  %78 : Tensor[] = aten::broadcast_tensors(%77)
  %79 : Tensor, %80 : Tensor, %81 : Tensor, %82 : Tensor = prim::ListUnpack(%78)
  %hy : Float(*, *), %cy : Float(*, *) = prim::FusionGroup_0(%cx, %82, %81, %80, %79)
  %30 : (Float(*, *), Float(*, *)) = prim::TupleConstruct(%hy, %cy)
  return (%30);

with prim::FusionGroup_0 = graph(%13 : Float(*, *),
      %71 : Tensor,
      %76 : Tensor,
      %81 : Tensor,
      %86 : Tensor):
  %87 : Float(*, *), %88 : Float(*, *), %89 : Float(*, *), %90 : Float(*, *) = prim::ConstantChunk[chunks=4, dim=1](%86)
  %82 : Float(*, *), %83 : Float(*, *), %84 : Float(*, *), %85 : Float(*, *) = prim::ConstantChunk[chunks=4, dim=1](%81)
  %77 : Float(*, *), %78 : Float(*, *), %79 : Float(*, *), %80 : Float(*, *) = prim::ConstantChunk[chunks=4, dim=1](%76)
  %72 : Float(*, *), %73 : Float(*, *), %74 : Float(*, *), %75 : Float(*, *) = prim::ConstantChunk[chunks=4, dim=1](%71)
  %69 : int = prim::Constant[value=1]()
  %70 : Float(*, *) = aten::add(%77, %72, %69)
  %66 : Float(*, *) = aten::add(%78, %73, %69)
  %62 : Float(*, *) = aten::add(%79, %74, %69)
  %58 : Float(*, *) = aten::add(%80, %75, %69)
  %54 : Float(*, *) = aten::add(%70, %82, %69)
  %50 : Float(*, *) = aten::add(%66, %83, %69)
  %46 : Float(*, *) = aten::add(%62, %84, %69)
  %42 : Float(*, *) = aten::add(%58, %85, %69)
  %38 : Float(*, *) = aten::add(%54, %87, %69)
  %34 : Float(*, *) = aten::add(%50, %88, %69)
  %30 : Float(*, *) = aten::add(%46, %89, %69)
  %26 : Float(*, *) = aten::add(%42, %90, %69)
  %ingate : Float(*, *) = aten::sigmoid(%38)
  %forgetgate : Float(*, *) = aten::sigmoid(%34)
  %cellgate : Float(*, *) = aten::tanh(%30)
  %outgate : Float(*, *) = aten::sigmoid(%26)
  %14 : Float(*, *) = aten::mul(%forgetgate, %13)
  %11 : Float(*, *) = aten::mul(%ingate, %cellgate)
  %cy : Float(*, *) = aten::add(%14, %11, %69)
  %4 : Float(*, *) = aten::tanh(%cy)
  %hy : Float(*, *) = aten::mul(%outgate, %4)
  return (%hy, %cy)
```


*Derivate Splitting* Many Graphs will require gradients (i.e. one of the inputs will have a `requires_grad`) property set. In this case, it is unsafe to run post-derivative optimizations directly on the Graph. Instead, our approach is to first *split* the Graph into sub-Graphs where symbolic gradient formulas are known and produce an explicit Graph for the forward pass along with a complementary Graph that implements the backwards pass using some of the values computed in the forward pass. We can then apply post-derivative optimization to the forward graph. The "gradOutputs" for the backwards graph are only known when the backward pass runs, so we cannot fully optimize it at this time. For instance, we do not know if some of those gradOutputs will also `require_grad` meaning that a gradient-of-gradient situation exists. Instead the backward pass will use a new GraphExecutor object to run and optimize its execution. In this way, we can handle an indefinite number of recursive gradient calculations.  
*Derivate Splitting* 许多图需要梯度属性设置（即其中一个输入有`requires_grad`） 。只这种情况下，直接在图上运行post-derivative优化是不安全的。相反，我们的方法首先将图拆分为子图，其中符号梯度公式是已知的，并且为forward pass生成一个显示图和一个使用forward pass计算的一些值实现backward pass的补充图，然后，我们可以将post-derivative优化应用于前向图,对于反向图的"gradOutputs" 只有在backward pass运行的时候才知道，所以这时我们不能完全优化它，例如，我们不知道一些gradOutputs也会需要`require_grad`，这意味着存在一个一个gradient-of-gradient。相反，backward pass将使用一个新的GraphExecutor对象运行和执行优化。用这种方式，我们可以处理不定数量的递归梯度计算。

The creating of derivative subgraphs is done using a similar approach to finding fusion groups: adjacent operations with known gradient formulas are grouped together into `prim::DifferentiableGraph` nodes. We only generate these nodes if we can find a large enough subgraph where optimization is likely to be profitable since there is some overhead involved in entering and exiting a differentiable subgraph.  
derivative子图的创建用一种类似于查找fusion group的方式完成：已知梯度公式的相邻操作被组合到`prim::DifferentiableGraph`节点中。如果我们找到一个足够大的字图，我们才会生成这些节点，这时优化才是有利可图的，因为进入和退出微分子图会涉及一些开销。

```
graph(%x : Float(*, *),
      %hx : Float(*, *),
      %cx : Float(*, *),
      %w_ih : Float(*, *),
      %w_hh : Float(*, *),
      %b_ih : Float(*),
      %b_hh : Float(*)):
  %8 : int = prim::Constant[value=1]()
  %hy : Float(*, *), %cy : Float(*, *) = prim::DifferentiableGraph_0(%cx, %b_hh, %b_ih, %hx, %w_hh, %x, %w_ih)
  %30 : (Float(*, *), Float(*, *)) = prim::TupleConstruct(%hy, %cy)
  return (%30)
with prim::DifferentiableGraph_0 = graph(%13 : Float(*, *),
      %29 : Float(*),
      %33 : Float(*),
      %40 : Float(*, *),
      %43 : Float(*, *),
      %45 : Float(*, *),
      %48 : Float(*, *)):
  %49 : Float(*, *) = aten::t(%48)
  %47 : Float(*, *) = aten::mm(%45, %49)
  %44 : Float(*, *) = aten::t(%43)
  %42 : Float(*, *) = aten::mm(%40, %44)
  %38 : int = prim::Constant[value=1]()
  %39 : Float(*, *) = aten::add(%47, %42, %38)
  %35 : Float(*, *) = aten::add(%39, %33, %38)
  %gates : Float(*, *) = aten::add(%35, %29, %38)
  %24 : Float(*, *), %25 : Float(*, *), %26 : Float(*, *), %27 : Float(*, *) = prim::ConstantChunk[chunks=4, dim=1](%gates)
  %ingate : Float(*, *) = aten::sigmoid(%24)
  %forgetgate : Float(*, *) = aten::sigmoid(%25)
  %cellgate : Float(*, *) = aten::tanh(%26)
  %outgate : Float(*, *) = aten::sigmoid(%27)
  %14 : Float(*, *) = aten::mul(%forgetgate, %13)
  %11 : Float(*, *) = aten::mul(%ingate, %cellgate)
  %cy : Float(*, *) = aten::add(%14, %11, %38)
  %4 : Float(*, *) = aten::tanh(%cy)
  %hy : Float(*, *) = aten::mul(%outgate, %4)
  return (%hy, %cy)
```

## JIT Logging ##

[jit_log.h](jit_log.h)

Logging is a very useful debugging technique, especially in the context of compilers. Compilers perform a series of passes and analyses and logging can help to trace issues such as wrong results or segmentation faults
all the way back to the original erroneous transformation.  
Logging是一个非常有用的调试技术，特别是在编译器的上下文中。编译器执行一系列的passes和analyses，日记可以帮助追踪问题，像错误结果或者分段错误，一直追溯到最初的错误转换。

`TorchScript` offers a simple logging facility that can enabled by setting an environment variable `PYTORCH_JIT_LOG_LEVEL`.  
`TorchScript`提供了一个简单的日志记录工具，可以通过设置环境变量`PYTORCH_JIT_LOG_LEVEL`来启用。

Logging is enabled on a per file basis. To enable logging in `dead_code_elimination.cpp`, `PYTORCH_JIT_LOG_LEVEL` should be
set to `dead_code_elimination.cpp` or, simply, to `dead_code_elimination` (i.e. `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination`).  
Logging是基于每个文件启用的。为了在`dead_code_elimination.cpp`中开启日志，`PYTORCH_JIT_LOG_LEVEL`应该被设置成`dead_code_elimination`（即`PYTORCH_JIT_LOG_LEVEL=dead_code_elimination`）

Multiple files can be logged by separating each file name with a colon `:` as in the following example, `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination:guard_elimination`  
通过用冒号分割每个文件名来记录多个文件，如下所示：  
`PYTORCH_JIT_LOG_LEVEL=dead_code_elimination:guard_elimination` 

There are 3 logging levels available for your use ordered by the detail level from lowest to highest.  
有3个日志级别可以使用，按详细的级别由低到高排序。

* `GRAPH_DUMP` should be used for printing entire graphs after optimization passes  
`GRAPH_DUMP` 用于在优化passes结束后打印整个图形
* `GRAPH_UPDATE` should be used for reporting graph transformations (i.e. node deletion, constant folding, etc)  
`GRAPH_UPDATE` 用于报告图形转换（如节点删除、常量折叠）
* `GRAPH_DEBUG` should be used for providing information useful for debugging
  the internals of a particular optimization pass or analysis  
  用于提供调试特定的优化pass或分析的有用信息

The current logging level is `GRAPH_UPDATE` meaning that both `GRAPH_DUMP` and `GRAPH_UPDATE` will be enabled when
one specifies a file(s) in `PYTORCH_JIT_LOG_LEVEL`.  
当前的日志级别是 `GRAPH_UPDATE`，这意味着 `PYTORCH_JIT_LOG_LEVEL`指定一个文件时，`GRAPH_DUMP`和`GRAPH_UPDATE`将被同时启用。

`GRAPH_DEBUG` can be enabled by prefixing a file name with an `>` as in `>alias_analysis`.
`>>` and `>>>` are also valid and **currently** are equivalent to `GRAPH_DEBUG` as there is no logging level that is
higher than `GRAPH_DEBUG`.  
`GRAPH_DEBUG`可以通过在文件名前加一个`>`来启用，例如`>alias_analysis`。`>>` 和`>>>`也是有效的，目前等同于`GRAPH_DEBUG`，因为没有比`GRAPH_DEBUG`更高的日记级别。

## DifferentiableGraphOp ##

[runtime/graph_executor.cpp](runtime/graph_executor.cpp)


A DifferentiableGraphOp combines an explicit forward Graph `f` with a paired backward graph `df`. When it runs, the input Tensors to `f` are detached from the autograd, the body of `f` is run, and then the autograd graph for the outputs of `f` are hooked up to the `df` function. The `df` function's outputs are also hooked up to the autograd graph.  
一个DifferentiableGraphOp将显示的前向图`f`与成对的后向图`df`相结合。当它运行时，`f`的输入Tensor将从autograd分离， 运行`f`的主体，然后`f`输出的augograd图将被挂到`df`函数上。`df`函数的输出也被挂到autograd图。

## Interpreter ##

* Code
* InterpreterState and interpreter design
* Fork/Wait

## FusionGroup ##

* inserted by passes

## Handling Mutability ##
### Aliasing and mutation in the PyTorch API
PyTorch API中的别名和变异  

In PyTorch, tensors are reference types. Operators can return "views" of the input tensor, creating a new tensor object that shares the same underlying storage as the original:  
在Pytorch中，tensor是引用类型。Operator可以返回输入张量的"views"，创建一个和原张量共享底层内存的新张量对象：

```python
a = torch.rand(2, 3)
b = a
# At this point, `a` and `b` share their storage.
c = b[0]
# `c` is shares storage with `a` and `b`, but only sees a slice of the allocated memory.
```

Some operators will *mutate* one or more of their operands in-place. These are typically denoted with a trailing underscore, or by taking an `out` argument as input:  
一些操作符会就地改变一个或多个它们的操作数，这些操作符一般尾部会有一个下划线，或者接受`out`参数作为输入：

```python
a = torch.zeros(2, 3)
b = torch.ones(2, 3)
a.add_(b)  # in-place add, so `a` is modified.
torch.add(a, b, out=a) # another way to express the same thing
```

### Aliasing and mutation annotations in FunctionSchema  
FunctionSchema中的别名和变异注释  
The JIT's `FunctionSchema`  allows operator writers to add annotations specifying the aliasing and mutation behavior of an operator. Optimization passes will use this information to determine whether transformations are semantics-preserving. This section provides a description of the alias annotation language, assuming that the reader already knows what `FunctionSchema` looks like.  
JIT的`FunctionSchema`允许操作符的作者添加注释指定运算符的别名和变异行为。优化passes将用该信息决定转换是否保留语义。本节提供了别名注释语言的描述，假设读者已经知道`FunctionSchema`是什么样子。

First, here is a pure function which always returns new memory:  
首先，这是一个返回新内存的pure函数
```
add(Tensor a, Tensor b) -> Tensor
```
The type `Tensor` with no annotations is sugar for "fresh, read-only `Tensor`". So since there are no annotations on anything, we know that this operator creates no aliases and mutates no inputs.  
不带注释的`Tensor`类型是“新鲜的，只读Tensor”的糖。由于没有任何注释，操作符不会创建别名和改变任何输入。

Next, a function that returns an alias to one of the inputs.:  
接下来，返回一个输入别名的函数

```
view(Tensor(a) self, int[] size) -> Tensor(a)
```
The shared `(a)` annotation on `self` and the output signify that the tensors will share the same storage. Another way to say is that `self` and the output belong to the same "alias set" `a`.  
在`self`上的共享`(a)`注释和输出tensor将共享相同的内存。另一种说法是`self`和输出属于相同的别名集`a`。

Now a function that writes in-place to one of the inputs (note the trailing underscore):  
就地修改输入的函数（注意尾部的下划线）：

```
add_(Tensor(a!) self, Tensor other) -> Tensor(a!)
```  
The `!` annotation means that this operator writes to the specified alias set (in this case `a`).  
`!`注释意味着操作符写入指定的别名集（这里是`a`）

Finally, sometimes we don't have enough information to provide an exact alias annotation. For example, here is the operator to extract an element from a list:  
最后，有时我们没有足够的信息提供准确的别名集。例如，从列表中提取一个元素的操作符：
```
list_select(Tensor[] list, int idx) -> Tensor(*)
```
Note the alias set `*`. This is the **wildcard set**. Optimization passes must assume that values in the wildcard set may alias any other value in the graph. This behavior is conservative and will disallow optimizations, but is guaranteed to be safe. In most cases, people shouldn't be writing operators with wildcard annotations. They are used as temporary workaround for when our alias analysis isn't sophisticated enough to understand something yet but we don't want to block feature development.  
注意别名集`*`是通配符集。优化passes必须假定通配符集中的值对齐图中的任何其它值，这种行为是保守的，并且不允许优化，但是保证是安全的。大多数情况下不允许写带有通配符注释的操作符，当我们的别名分析不够复杂无法理解某些内容，但我们又不想阻止特性开发，它们被当做临时方案。

This annotation language is consumed by the `FunctionSchema` parser, which produces `AliasInfo` objects summarizing the aliasing relationships for each schema `Argument`.    
注释语言由`FunctionSchema` parser使用，它会生成`AliasInfo`对象，总结每个模式`Argument`的别名关系。

### Alias Analysis in the IR
[ir/alias_analysis.h](ir/alias_analysis.h)
An alias analysis pass consumes the per-operator aliasing information to construct a database of aliasing and mutation relationships in a graph, called `AliasDb`. This section focuses on the alias analysis pass; the public interface to `AliasDb` will be described later.  
别名分析pass使用每个运算符的别名信息在图中构造一个别名和变异关系的数据库，称为`AliasDb`。这一部分聚焦在别名分析pass;`AliasDb`public接口将在后面描述。

The core data structure in the AliasDb is called `AliasTracker`, which is a DAG where the edges are "may point to" relationships and the  vertices are aliasing `Element`s. The most common kind of `Element` is an IR `Value`, but there are other kinds of things that can alias that aren't first-class `Value`s in the IR, like wildcards or contained types (such as in a list or tuple).  
AliasDb中核心的数据结构是`AliasTracker`，它是一个DAG，其中的边是可能指向关系，顶点是别名`Element`。`Element`大部分类型是IR `Value`，但是其它的类型也可以别名，它们不是IR中的first-class `Value`，像通配符或者包含类型（像list或tuple）。

The alias analysis pass walks through the nodes in a graph, examining schema `AliasInfo`  objects and adding edges in the `AliasTracker` DAG accordingly. For example, for the node:  
别名分析pass遍历图中的节点，检查模式`AliasInfo`对象，并相应的在`AliasTracker`DAG中添加边。例如，对于节点：

```
%output : Tensor = aten::view(%self, %size)
```
the analyzer will examine the schema for `view()`:  
分析器将检查`view()`的模式
```
view(Tensor(a) self, int[] size) -> Tensor(a)
```
and add an edge from `%output` to `%self`. The alias analysis pass is flow-insensitive, as we are only adding "points-to" edges when processing a node.  
并添加一条从`%output`到`%self`的边，别名分析pass是flow-insensitive，因为只有当处理节点时才添加指向边。

As a more involved example, the following TorchScript snippet:  
一个更复杂的示例，如下的TorchScript片段：  

```python
@torch.jit.script
def foo(a : Tensor, b : Tensor):
        c = 2 * b
  a += 1
  if a.max() > 4:
    r = a[0]
  else:
    r = b[0]
  return c, r
```
Will produce a graph like this:  
会产生如下这样的图:  
![AliasTracker graph](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/aliastracker_graph.png)    


A few things to note:  
需要注意：
- "Graph Input Element" is an example of an `Element` that isn't a first-class `Value`. Alias analysis happens on a per-function level, so we don't necessarily know the aliasing relationships of the inputs. The only safe assumption is that `a` and `b` may alias each other, so they point to a special `Element` that describes "the world outside of this function".  
"Graph Input Element"是一个`Element`的示例，不是first-class`Value`。别名分析在每个函数级别发生，因此我们不知道输入的别名关系。唯一安全的假设是`a`和`b`互为别名，因此它们指向一个特殊的`Element`来描述“函数外面的世界”。

- `r` may point to either `a` or `b`, depending on the runtime value of `a.max()`.  A given `Element` may point to multiple other `Element`s. This can happen if there is branching control flow (like in this example), or with certain ops like `contiguous()`, which either returns an alias to the input or a fresh Tensor, depending on the runtime characteristics of the input.      
`r`可能指向`a`或者`b`，这取决于`a.max()`运行时的值，一个`Element`可能指向多个其它`Element`。如果存在分支控制流（像本例）或者某些操作符像`contiguous()`，会根据输入运行时特性，返回输入别名或者一个新的Tensor。

- `c` is a fresh tensor (i.e. it doesn't point to anything) since it was created using the pure operation `2 * b`.  
`c`是一个新的tensor（它不指向任何东西），因为它使用一个pure操作符`2*b`创建。

The last point demonstrates a key concept: *leaf elements uniquely describe memory locations*. Since a leaf element doesn't point to anything, the memory that backs it must have been freshly allocated by some op. Thus we can use leaf elements to represent disjoint memory locations.  
最后一点证明了一个关键概念：叶子元素描述了唯一的内存位置。由于叶子元素不指向任何东西，它的内存是由某些操作符新分配的，因此，我们可以用叶子元素表示不相交的内存位置。

So to determine whether  `a` and `b` may alias, we traverse the `AliasTracker` DAG and figure out if `a` and `b` share any leaf nodes. If they do, then we know `a` and `b` might point to the same memory location, i.e. `a` and `b` may alias. This kind of query is common enough that `AliasTracker` does path compression to speed up leaf-finding, so that aliasing queries can be serviced in amortized constant time.   
因此，为了确定`a`和`b`可能存在别名，我们遍历`AliasTracker` DAG并找出`a`和`b`是否共享叶子节点。如果有，那么`a`和`b`可能指向相同的内存位置，也就是说`a`和`b`可能有别名。这种类型的查询是常见的，`AliasTracker`会进行路径压缩来加速叶子节点查找，因此可以在平摊常数时间内处理别名查询。

### Writing optimization passes with `AliasDb`  
使用`AliasDb`的写入优化passes  

`AliasDb` provides a high-level interface to help people write mutability-safe optimization passes.  
`AliasDb`提供了一个高级接口来帮助我们写可变安全的优化passes。

In particular, `moveAfterTopologicallyValid()` (and it's `moveBefore` variant) will reorder nodes in a way that preserves data dependencies and avoids any data hazards.  The rules for this are that all mutable *writes* to a given memory location must occur in the same order (avoid WAW hazards), and that no reads can be reordered before or after any write (WAR, RAW hazards).  
特别是，`moveAfterTopologicallyValid()`（它是`moveBefore`的变体）以保留数据依赖性和避免任何数据危害的方式重新排序节点。这方面的规则是，对于给定内存的所有可变写入必须以相同的顺序执行（避免WAW危害），并且任何读操作都不能在写入之前或之后重新排序（WAR、RAW 危害）

However, reordering of reads across writes *is allowed* if we can prove that the read cannot alias the thing being written. This happens whenever we have tensors that come from functions that produce fresh results (common) inside of the function. It also happens whenever the creation of the mutable tensor is seen in the function (so it gets assigned a fresh variable), and all of its writes occur in that function.   
然而，如果我们能证明读取不能为正在写入的东西设置别名，那么跨写入重新排序读取是被允许的。这发生在来自函数张量在函数内部产生新的结果，这也发生在函数内部可变张量的创建（因此它被赋予一个新的变量），并且它的所有写入也发生在该函数。

The intention is that if you only mutate the graph through `AliasDb`, you don't have to think about mutability/aliasing at all in your pass. As we write more passes, the interface to `AliasDb` will get richer (one example is transforming an in-place operation to its pure equivalent if we can prove it's safe).  
这样做的目的是如果你仅仅通过`AliasDb`改变图，在你的所有pass中完全不需要考虑mutability/aliasing，随着我们的passes增多，`AliasDb`接口将变得更加丰富（一个例子是如果我们证明它是安全的，就地操作转换为它的纯等效操作）

`AliasDb` also provides lower level APIs that users of LLVM's alias analysis pass would be familiar with, such as querying whether any two `Value`s may alias.  
`AliasDb`也提供低级别的APIs，LLVM的别名分析pass用户将熟悉它，像查询任意两个`Value`是否可以别名

TODO: differentiation, symbolic autograd,  
TODO：微分，符号autograd   
TODO: fusion, operators  
TODO：融合，运算符


# Profiling Programs

`prim::profile` nodes are inserted on every **use** of a value by `ProfilingRecord::instrumentBlock`. Every `prim::profile` node runs a lambda that uses a captured, initial type value and the type of an incoming tensor and merges the two into a refined `TensorType`  
`ProfilingRecord::instrumentBlock`每次使用一个值时插入`prim::profile`节点，每个`prim::profile`节点运行一个lambda，它捕获初始类型知和输入张量类型并将它们合并成一个细化的`TensorType`

`prim::profile` nodes are replaced with `prim::Guard` nodes by `InsertGuards`. `prim::Guard` nodes are inserted to guarantee that beyond the guard a guarded tensor will always be of the profiled shape. This guarantee will enable optimizations and codegens to generate more efficient code.  
`InsertGuards`用`prim::Guard`节点替换`prim::profile`。插入`prim::Guard`节点为了保证在保护之外的受保护张量始终具有轮廓形状。这种保护使得优化和代码生成器生成更加高效的代码。

JIT attempts to reduce the number of `prim::Guard` nodes as these nodes may interefere with optimizations.    
JIT 试图减少`prim::Guard`节点的数量，因为这些节点可能会干扰优化。

* First, `GuardElimination::moveGuardsToDefs` tries to move `prim::Guards` to their definitions, so the guards guarding the same tensor follow the definition directly or another guard on the same tensor. This step is done in  
首先，`GuardElimination::moveGuardsToDefs`尝试将`prim::Guards`move到它们的定义中，因此保护同一张量的守卫直接遵循定义或者同一张量上的另一守卫。这一步是在

* This ordering allows us to **coalesce** (done in `GuardElimination::coalesceGuards`) multiple guards into a single one.  
这种排序允许我们合并（在`GuardElimination::coalesceGuards`完成）多个守卫为一个  

* After guards are  **coaslesced** , `GuardElimination::eliminateGuards` attempts to eliminate more guards as follows: it inspects each operation and its inputs. It checks if inputs to the operation are guarded and also if the operation produces the consistent shapes given the guarded inputs. For example, if two inputs to `add` are guaranteed to be of shape `(2, 3) `, the output shape will also always be `(2, 3)` If this property holds, JIT is allowed to remove the guard guarding operation's output.  
所有的守卫被合并后，`GuardElimination::eliminateGuards`试图消除更多的守卫，如下：它检查每个操作以及它们的输入。它检查操作的输入是否受保护和在给定受保护输入的情况下操作符是否产生一致形状的输出。例如，如果`add`的两个输入保证为形状`(2, 3) `，输出形状将总是`(2, 3) `，如果这个属性保持不变，JIT允许移除保护操作的输出。

Lastly, JIT needs to be handle cases when the assumptions about tensor shapes fail at runtime. To handle guard failures, JIT needs to be able to run the original code i.e. the code  that doesn't rely on assumptions about shapes. As guards can be inserted and moved (by Optimizer) at/to arbitrary points in a computional graph, JIT needs to be able to resume execution starting from those arbitrary points onward.  
最后，当关于张量形状的假设在运行时失败，JIT需要处理。为了处理保护失败，JIT需要能够运行原始code，即code不依赖于形状假设。因为可以在图中的任意点插入和移动（通过Optimizer）守卫，JIT需要能够从任意点重新开始执行。

`InsertBailoutNodes` builds deoptimized versions of the original computational graph, that contain the rest of computations starting from their corresponding guard failure poins and, also, captures live values needed to execute those deoptimized graphs. In other words, the pass replaces `prim::Guard` nodes with `prim::BailOut` nodes which have the`attr::Subgraph` attributes set to the deoptimized versions of the  remaining computations at their corresponding `prim::Guard`s.   
`InsertBailoutNodes`构建原始计算图的非优化版本，它包含了从保护故障点开始的其余计算，并且捕获非优化图执行需要的活值。换句话说，这个pass将`prim::Guard`节点替换为`prim::BailOut`节点，这些节点将`attr::Subgraph`属性设置为对应的`prim::Guard`节点剩余计算的非优化版本。

# Saving Programs

See [the serialization docs](docs/serialization.md).

# Testing Programs
## Test Autodiff ##
[runtime/symbolic_script.cpp](runtime/symbolic_script.cpp)

When differentiating a graph, each node that has a symbolic gradient will be included in a `prim::DifferentiableGraph`. We fall back to use autograd for the node if there isn't a gradient formula for it.
Adding/updating symbolic gradient functions must be tested carefully as it's easy to get CI green by comparing autograd result with itself, but potentially cause autodiff support regression.  
当微分一个图，具有符号梯度的节点将被包含在`prim::DifferentiableGraph`。如果节点对于符号梯度没有梯度公式，那么节点将回退到使用autograd，添加/更新符号梯度函数需要仔细测试，因为通过比较autograd结果，很容易得到CI green，但是可能造成autodiff支持回归。

If your PR adds/updates a gradient formula for `torch`/`nn` functions, you **MUST** enable/update the corresponding tests in  
如果你的PR对`torch`/`nn`函数添加/更新梯度公式，则必须启用/更新相应的测试
- `torch` functions: `method_tests` in [common_method_tests.py](../../../test/common_method_tests.py)
- `nn` functions: `nn_functional_tests` in [test_jit.py](../../../test/test_jit.py)

To turn on autodiff check, you can add an optional `check_ad(should_check_autodiff[bool], nonfusible_nodes[str|list[str]], fusible_nodes[str|list[str]])` tuple after the optional test variant name field.  
为了开启autodiff检查，你可以在可选的test variant字段后面添加一个可选的`check_ad(should_check_autodiff[bool], nonfusible_nodes[str|list[str]], fusible_nodes[str|list[str]])` 元组。

If `should_check_autodiff=True`, the differentiated traced/script forward graph must have a `prim::DifferentiableGraph`.  
All nodes in `nonfusible_nodes` should show up in at least once in `prim::DifferentiableGraph` subgraphs.
When fusion is enabled, all nodes in `fusible_nodes` should show up in one of `prim::FusionGroup` graphs attached to `prim::DifferentiableGraph`,
otherwise they're checked as `nonfusible_nodes` as well.  
如果`should_check_autodiff=True`, 微分的traced/script前向图必须有一个`prim::DifferentiableGraph`。
在`nonfusible_nodes`中的所有节点在`prim::DifferentiableGraph`子图中至少显示一次。当开启融合，`fusible_nodes`中的所有节点都应该在`prim::FusionGroup`图中显示，`prim::FusionGroup`图附加到`prim::DifferentiableGraph`，否则，它们被检查为`nonfusible_nodes`。  
On the other hand, if `should_check_autodiff=False`, the graph can still have `prim::DifferentiableGraph` with other nodes, but not `nonfusible_nodes` and `fusible_nodes`.  
另一方面，如果`should_check_autodiff=False`，图仍然可以有`prim::DifferentiableGraph`，但是没有`nonfusible_nodes`和 `fusible_nodes`。

To make writing test easier, you only need to write out node names if it's different from the function name. Below are a few examples:  
为了使测试更容易，你只需要写出与函数名不同的节点名。下面是几个例子;
```python
('conv1d', ...), # No symbolic gradient formula
('avg_pool2d', ..., (True,)), # Has symbolic gradient formula, only has one nonfusible node aten::avg_pool2d
('nll_loss', ..., (True, 'aten::nll_loss_forward')), # Is replaced by a different node in its symbolic gradient formula
('dropout', ..., (True, ['prim::is_cuda', 'aten::bernoulli_'], ['aten::rand_like', ..., 'aten::div'])), # Some op are fused when fusion is enabled.
```

Note that even for the same function, different tests could trigger different function schemas (e.g `aten::add`) while only a few of them have symbolic gradient formulas.
You should only turn on autodiff check in tests who have symbolic gradient. If you are not sure, uncomment the debugging line in [runtime/symbolic_script.cpp](runtime/symbolic_script.cpp)
to check which function schema the test triggers.  
注意即使是相同的函数，不同的测试可以触发不同的函数schemas（例如`aten::add`），而只有少数函数有符号梯度公式。你应该在具有符号梯度的测试中打开autodiff检查，如果你不确定是否具有符号梯度，可以在 [runtime/symbolic_script.cpp](runtime/symbolic_script.cpp)注释掉debugging行，以检查测试触发的函数schema。


## Python Printer

[serialization/python_print.cpp](serialization/python_print.cpp)
[serialization/import_source.cpp](serialization/import_source.cpp)

The Python Printer takes a `Graph` and produces Python-like code that represents the same graph. Using some special values in [serialization/import_source.cpp](serialization/import_source.cpp), this code can be read back in by the compiler to produce the same `Graph`. In Python a `ScriptModule`'s `code` property shows the Python Printed graph.    
Python打印器接受一个`Graph`并生成同一个图的类python code。使用一些在[serialization/import_source.cpp](serialization/import_source.cpp)中的特殊值，编译器可以读取code并产生相同的图。在python中`ScriptModule`的`code`属性显示为python打印图。

The table below shows the graph and code for this small `ScriptModule`:  
下表显示了`ScriptModule`的graph和code  

```python
class M(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x, y, z):
        # type: (Tensor, int, float) -> Tensor
        if y > 2:
            x = x + z
        else:
            x = x + y
        return x

m = M()
```

`m.graph`
```
graph(%x.1 : Tensor,
      %y : int,
      %z : float):
  %5 : int = prim::Constant[value=1]()
  %3 : int = prim::Constant[value=2]()
  %4 : bool = aten::gt(%y, %3)
  %x : Tensor = prim::If(%4)
    block0():
      %x.2 : Tensor = aten::add(%x.1, %z, %5)
      -> (%x.2)
    block1():
      %x.3 : Tensor = aten::add(%x.1, %y, %5)
      -> (%x.3)
  return (%x)
```

`m.code`
```python
def forward(self,
    x: Tensor,
    y: int,
    z: float) -> Tensor:
  if torch.gt(y, 2):
    x0 = torch.add(x, z, 1)
  else:
    x0 = torch.add(x, y, 1)
  return x0
```

# Python Bindings

TODO: Script Module, torch.jit.trace, __constant__ handling, weak script modules  
