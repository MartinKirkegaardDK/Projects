��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
h
Any	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListReserve
element_shape"
shape_type
num_elements(
handle���element_dtype"
element_dtypetype"

shape_typetype:
2	
�
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint���������
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
�
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
�
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:	*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:	*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
%simple_rnn_5/simple_rnn_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*6
shared_name'%simple_rnn_5/simple_rnn_cell_5/kernel
�
9simple_rnn_5/simple_rnn_cell_5/kernel/Read/ReadVariableOpReadVariableOp%simple_rnn_5/simple_rnn_cell_5/kernel*
_output_shapes
:	�*
dtype0
�
/simple_rnn_5/simple_rnn_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*@
shared_name1/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel
�
Csimple_rnn_5/simple_rnn_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel*
_output_shapes

:*
dtype0
�
#simple_rnn_5/simple_rnn_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#simple_rnn_5/simple_rnn_cell_5/bias
�
7simple_rnn_5/simple_rnn_cell_5/bias/Read/ReadVariableOpReadVariableOp#simple_rnn_5/simple_rnn_cell_5/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

:	*
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:	*
dtype0
�
,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*=
shared_name.,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m
�
@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m*
_output_shapes
:	�*
dtype0
�
6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m
�
JAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m*
_output_shapes

:*
dtype0
�
*Adam/simple_rnn_5/simple_rnn_cell_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m
�
>Adam/simple_rnn_5/simple_rnn_cell_5/bias/m/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

:	*
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:	*
dtype0
�
,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*=
shared_name.,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v
�
@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v*
_output_shapes
:	�*
dtype0
�
6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*G
shared_name86Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v
�
JAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v*
_output_shapes

:*
dtype0
�
*Adam/simple_rnn_5/simple_rnn_cell_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v
�
>Adam/simple_rnn_5/simple_rnn_cell_5/bias/v/Read/ReadVariableOpReadVariableOp*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�.
value�.B�. B�.
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses*
�
#iter

$beta_1

%beta_2
	&decay
'learning_ratemXmY(mZ)m[*m\v]v^(v_)v`*va*
'
(0
)1
*2
3
4*
'
(0
)1
*2
3
4*
* 
�
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

0serving_default* 
* 
* 
* 
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
�

(kernel
)recurrent_kernel
*bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:_random_generator
;__call__
*<&call_and_return_all_conditional_losses*
* 

(0
)1
*2*

(0
)1
*2*
* 
�

=states
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%simple_rnn_5/simple_rnn_cell_5/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#simple_rnn_5/simple_rnn_cell_5/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

H0
I1*
* 
* 
* 
* 
* 
* 
* 
* 

(0
)1
*2*

(0
)1
*2*
* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
6	variables
7trainable_variables
8regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
8
	Ototal
	Pcount
Q	variables
R	keras_api*
H
	Stotal
	Tcount
U
_fn_kwargs
V	variables
W	keras_api*
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

O0
P1*

Q	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

S0
T1*

V	variables*
�{
VARIABLE_VALUEAdam/dense_5/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/simple_rnn_5/simple_rnn_cell_5/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_5/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_5/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE*Adam/simple_rnn_5/simple_rnn_cell_5/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_masking_5_inputPlaceholder*5
_output_shapes#
!:�������������������*
dtype0**
shape!:�������������������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_masking_5_input%simple_rnn_5/simple_rnn_cell_5/kernel#simple_rnn_5/simple_rnn_cell_5/bias/simple_rnn_5/simple_rnn_cell_5/recurrent_kerneldense_5/kerneldense_5/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_58516
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9simple_rnn_5/simple_rnn_cell_5/kernel/Read/ReadVariableOpCsimple_rnn_5/simple_rnn_cell_5/recurrent_kernel/Read/ReadVariableOp7simple_rnn_5/simple_rnn_cell_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m/Read/ReadVariableOpJAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m/Read/ReadVariableOp>Adam/simple_rnn_5/simple_rnn_cell_5/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp@Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v/Read/ReadVariableOpJAdam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v/Read/ReadVariableOp>Adam/simple_rnn_5/simple_rnn_cell_5/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_59205
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%simple_rnn_5/simple_rnn_cell_5/kernel/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel#simple_rnn_5/simple_rnn_cell_5/biastotalcounttotal_1count_1Adam/dense_5/kernel/mAdam/dense_5/bias/m,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m*Adam/simple_rnn_5/simple_rnn_cell_5/bias/mAdam/dense_5/kernel/vAdam/dense_5/bias/v,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v6Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_59287��
�,
�
while_body_58942
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:	�G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_5_matmul_readvariableop_resource:	�E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������w
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���}
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_58069

inputs%
simple_rnn_5_58056:	� 
simple_rnn_5_58058:$
simple_rnn_5_58060:
dense_5_58063:	
dense_5_58065:	
identity��dense_5/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
masking_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_masking_5_layer_call_and_return_conditional_losses_57721�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall"masking_5/PartitionedCall:output:0simple_rnn_5_58056simple_rnn_5_58058simple_rnn_5_58060*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58020�
dense_5/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0dense_5_58063dense_5_58065*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_57869�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	�
NoOpNoOp ^dense_5/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�c
�
!__inference__traced_restore_59287
file_prefix1
assignvariableop_dense_5_kernel:	-
assignvariableop_1_dense_5_bias:	&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: K
8assignvariableop_7_simple_rnn_5_simple_rnn_cell_5_kernel:	�T
Bassignvariableop_8_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel:D
6assignvariableop_9_simple_rnn_5_simple_rnn_cell_5_bias:#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: ;
)assignvariableop_14_adam_dense_5_kernel_m:	5
'assignvariableop_15_adam_dense_5_bias_m:	S
@assignvariableop_16_adam_simple_rnn_5_simple_rnn_cell_5_kernel_m:	�\
Jassignvariableop_17_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_m:L
>assignvariableop_18_adam_simple_rnn_5_simple_rnn_cell_5_bias_m:;
)assignvariableop_19_adam_dense_5_kernel_v:	5
'assignvariableop_20_adam_dense_5_bias_v:	S
@assignvariableop_21_adam_simple_rnn_5_simple_rnn_cell_5_kernel_v:	�\
Jassignvariableop_22_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_v:L
>assignvariableop_23_adam_simple_rnn_5_simple_rnn_cell_5_bias_v:
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp8assignvariableop_7_simple_rnn_5_simple_rnn_cell_5_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpBassignvariableop_8_simple_rnn_5_simple_rnn_cell_5_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp6assignvariableop_9_simple_rnn_5_simple_rnn_cell_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_5_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_5_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp@assignvariableop_16_adam_simple_rnn_5_simple_rnn_cell_5_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpJassignvariableop_17_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp>assignvariableop_18_adam_simple_rnn_5_simple_rnn_cell_5_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_5_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_5_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp@assignvariableop_21_adam_simple_rnn_5_simple_rnn_cell_5_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpJassignvariableop_22_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_simple_rnn_5_simple_rnn_cell_5_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
while_cond_57471
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_57471___redundant_placeholder03
/while_while_cond_57471___redundant_placeholder13
/while_while_cond_57471___redundant_placeholder23
/while_while_cond_57471___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�,
�
while_body_58834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:	�G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_5_matmul_readvariableop_resource:	�E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������w
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���}
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�|
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_58333

inputsP
=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource:	�L
>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource:Q
?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource:;
)dense_5_tensordot_readvariableop_resource:	5
'dense_5_biasadd_readvariableop_resource:	
identity��dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp�4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp�6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp�simple_rnn_5/whileY
masking_5/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
masking_5/NotEqualNotEqualinputsmasking_5/NotEqual/y:output:0*
T0*5
_output_shapes#
!:�������������������j
masking_5/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
masking_5/AnyAnymasking_5/NotEqual:z:0(masking_5/Any/reduction_indices:output:0*4
_output_shapes"
 :������������������*
	keep_dims(|
masking_5/CastCastmasking_5/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :������������������p
masking_5/mulMulinputsmasking_5/Cast:y:0*
T0*5
_output_shapes#
!:��������������������
masking_5/SqueezeSqueezemasking_5/Any:output:0*
T0
*0
_output_shapes
:������������������*
squeeze_dims

���������S
simple_rnn_5/ShapeShapemasking_5/mul:z:0*
T0*
_output_shapes
:j
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:���������p
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose	Transposemasking_5/mul:z:0$simple_rnn_5/transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������^
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
simple_rnn_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_5/ExpandDims
ExpandDimsmasking_5/Squeeze:output:0$simple_rnn_5/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :������������������r
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose_1	Transpose simple_rnn_5/ExpandDims:output:0&simple_rnn_5/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :������������������s
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%simple_rnn_5/simple_rnn_cell_5/MatMulMatMul%simple_rnn_5/strided_slice_2:output:0<simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0=simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
'simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMulsimple_rnn_5/zeros:output:0>simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"simple_rnn_5/simple_rnn_cell_5/addAddV2/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:01simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:����������
#simple_rnn_5/simple_rnn_cell_5/TanhTanh&simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������{
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : u
*simple_rnn_5/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_5/TensorArrayV2_2TensorListReserve3simple_rnn_5/TensorArrayV2_2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
Dsimple_rnn_5/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6simple_rnn_5/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose_1:y:0Msimple_rnn_5/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:���
simple_rnn_5/zeros_like	ZerosLike'simple_rnn_5/simple_rnn_cell_5/Tanh:y:0*
T0*'
_output_shapes
:���������p
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros_like:y:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fsimple_rnn_5/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :���������:���������: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
simple_rnn_5_while_body_58225*)
cond!R
simple_rnn_5_while_cond_58224*M
output_shapes<
:: : : : :���������:���������: : : : : : *
parallel_iterations �
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0u
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskr
simple_rnn_5/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose_2	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :�������������������
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:	*
dtype0`
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_5/Tensordot/ShapeShapesimple_rnn_5/transpose_2:y:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_5/Tensordot/transpose	Transposesimple_rnn_5/transpose_2:y:0!dense_5/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	c
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	a
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������	�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������	s
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :������������������	u
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*4
_output_shapes"
 :������������������	�
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp6^simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5^simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp^simple_rnn_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2n
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2l
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�K
�
simple_rnn_5_while_body_582256
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_2$
 simple_rnn_5_while_placeholder_35
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0u
qsimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor_0X
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0:	�T
Fsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:Y
Gsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:
simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_4!
simple_rnn_5_while_identity_53
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensors
osimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensorV
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource:	�R
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource:W
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource:��;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp�<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
Fsimple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
8simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemqsimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor_0simple_rnn_5_while_placeholderOsimple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
+simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
,simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAdd5simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Csimple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
-simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul simple_rnn_5_while_placeholder_3Dsimple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(simple_rnn_5/while/simple_rnn_cell_5/addAddV25simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:07simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:����������
)simple_rnn_5/while/simple_rnn_cell_5/TanhTanh,simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������r
!simple_rnn_5/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
simple_rnn_5/while/TileTile?simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem:item:0*simple_rnn_5/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
simple_rnn_5/while/SelectV2SelectV2 simple_rnn_5/while/Tile:output:0-simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0 simple_rnn_5_while_placeholder_2*
T0*'
_output_shapes
:���������t
#simple_rnn_5/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
simple_rnn_5/while/Tile_1Tile?simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem:item:0,simple_rnn_5/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
simple_rnn_5/while/SelectV2_1SelectV2"simple_rnn_5/while/Tile_1:output:0-simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0 simple_rnn_5_while_placeholder_3*
T0*'
_output_shapes
:����������
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1simple_rnn_5_while_placeholder$simple_rnn_5/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:���Z
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: �
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: :����
simple_rnn_5/while/Identity_4Identity$simple_rnn_5/while/SelectV2:output:0^simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:����������
simple_rnn_5/while/Identity_5Identity&simple_rnn_5/while/SelectV2_1:output:0^simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:����������
simple_rnn_5/while/NoOpNoOp<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"G
simple_rnn_5_while_identity_5&simple_rnn_5/while/Identity_5:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"�
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"�
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"�
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"�
osimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensorqsimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor_0"�
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :���������:���������: : : : : : 2z
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2x
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2|
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
� 
�
while_body_57472
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_simple_rnn_cell_5_57494_0:	�-
while_simple_rnn_cell_5_57496_0:1
while_simple_rnn_cell_5_57498_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_5_57494:	�+
while_simple_rnn_cell_5_57496:/
while_simple_rnn_cell_5_57498:��/while/simple_rnn_cell_5/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
/while/simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_5_57494_0while_simple_rnn_cell_5_57496_0while_simple_rnn_cell_5_57498_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_57459�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity8while/simple_rnn_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������~

while/NoOpNoOp0^while/simple_rnn_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_5_57494while_simple_rnn_cell_5_57494_0"@
while_simple_rnn_cell_5_57496while_simple_rnn_cell_5_57496_0"@
while_simple_rnn_cell_5_57498while_simple_rnn_cell_5_57498_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2b
/while/simple_rnn_cell_5/StatefulPartitionedCall/while/simple_rnn_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
,__inference_simple_rnn_5_layer_call_fn_58554
inputs_0
unknown:	�
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_57694|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�=
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58900

inputsC
0simple_rnn_cell_5_matmul_readvariableop_resource:	�?
1simple_rnn_cell_5_biasadd_readvariableop_resource:D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������k
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_58834*
condR
while_cond_58833*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
 __inference__wrapped_model_57411
masking_5_input]
Jsequential_5_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource:	�Y
Ksequential_5_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource:^
Lsequential_5_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource:H
6sequential_5_dense_5_tensordot_readvariableop_resource:	B
4sequential_5_dense_5_biasadd_readvariableop_resource:	
identity��+sequential_5/dense_5/BiasAdd/ReadVariableOp�-sequential_5/dense_5/Tensordot/ReadVariableOp�Bsequential_5/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp�Asequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp�Csequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp�sequential_5/simple_rnn_5/whilef
!sequential_5/masking_5/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_5/masking_5/NotEqualNotEqualmasking_5_input*sequential_5/masking_5/NotEqual/y:output:0*
T0*5
_output_shapes#
!:�������������������w
,sequential_5/masking_5/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
sequential_5/masking_5/AnyAny#sequential_5/masking_5/NotEqual:z:05sequential_5/masking_5/Any/reduction_indices:output:0*4
_output_shapes"
 :������������������*
	keep_dims(�
sequential_5/masking_5/CastCast#sequential_5/masking_5/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :�������������������
sequential_5/masking_5/mulMulmasking_5_inputsequential_5/masking_5/Cast:y:0*
T0*5
_output_shapes#
!:��������������������
sequential_5/masking_5/SqueezeSqueeze#sequential_5/masking_5/Any:output:0*
T0
*0
_output_shapes
:������������������*
squeeze_dims

���������m
sequential_5/simple_rnn_5/ShapeShapesequential_5/masking_5/mul:z:0*
T0*
_output_shapes
:w
-sequential_5/simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential_5/simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential_5/simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'sequential_5/simple_rnn_5/strided_sliceStridedSlice(sequential_5/simple_rnn_5/Shape:output:06sequential_5/simple_rnn_5/strided_slice/stack:output:08sequential_5/simple_rnn_5/strided_slice/stack_1:output:08sequential_5/simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential_5/simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
&sequential_5/simple_rnn_5/zeros/packedPack0sequential_5/simple_rnn_5/strided_slice:output:01sequential_5/simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%sequential_5/simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
sequential_5/simple_rnn_5/zerosFill/sequential_5/simple_rnn_5/zeros/packed:output:0.sequential_5/simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:���������}
(sequential_5/simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
#sequential_5/simple_rnn_5/transpose	Transposesequential_5/masking_5/mul:z:01sequential_5/simple_rnn_5/transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������x
!sequential_5/simple_rnn_5/Shape_1Shape'sequential_5/simple_rnn_5/transpose:y:0*
T0*
_output_shapes
:y
/sequential_5/simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_5/simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_5/simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)sequential_5/simple_rnn_5/strided_slice_1StridedSlice*sequential_5/simple_rnn_5/Shape_1:output:08sequential_5/simple_rnn_5/strided_slice_1/stack:output:0:sequential_5/simple_rnn_5/strided_slice_1/stack_1:output:0:sequential_5/simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
(sequential_5/simple_rnn_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
$sequential_5/simple_rnn_5/ExpandDims
ExpandDims'sequential_5/masking_5/Squeeze:output:01sequential_5/simple_rnn_5/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :������������������
*sequential_5/simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
%sequential_5/simple_rnn_5/transpose_1	Transpose-sequential_5/simple_rnn_5/ExpandDims:output:03sequential_5/simple_rnn_5/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :�������������������
5sequential_5/simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
'sequential_5/simple_rnn_5/TensorArrayV2TensorListReserve>sequential_5/simple_rnn_5/TensorArrayV2/element_shape:output:02sequential_5/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Osequential_5/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
Asequential_5/simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor'sequential_5/simple_rnn_5/transpose:y:0Xsequential_5/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���y
/sequential_5/simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential_5/simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential_5/simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)sequential_5/simple_rnn_5/strided_slice_2StridedSlice'sequential_5/simple_rnn_5/transpose:y:08sequential_5/simple_rnn_5/strided_slice_2/stack:output:0:sequential_5/simple_rnn_5/strided_slice_2/stack_1:output:0:sequential_5/simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
Asequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpJsequential_5_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
2sequential_5/simple_rnn_5/simple_rnn_cell_5/MatMulMatMul2sequential_5/simple_rnn_5/strided_slice_2:output:0Isequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Bsequential_5/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpKsequential_5_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
3sequential_5/simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd<sequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0Jsequential_5/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Csequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpLsequential_5_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
4sequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMul(sequential_5/simple_rnn_5/zeros:output:0Ksequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/sequential_5/simple_rnn_5/simple_rnn_cell_5/addAddV2<sequential_5/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:0>sequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:����������
0sequential_5/simple_rnn_5/simple_rnn_cell_5/TanhTanh3sequential_5/simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:����������
7sequential_5/simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
)sequential_5/simple_rnn_5/TensorArrayV2_1TensorListReserve@sequential_5/simple_rnn_5/TensorArrayV2_1/element_shape:output:02sequential_5/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���`
sequential_5/simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : �
7sequential_5/simple_rnn_5/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
)sequential_5/simple_rnn_5/TensorArrayV2_2TensorListReserve@sequential_5/simple_rnn_5/TensorArrayV2_2/element_shape:output:02sequential_5/simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
Qsequential_5/simple_rnn_5/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Csequential_5/simple_rnn_5/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor)sequential_5/simple_rnn_5/transpose_1:y:0Zsequential_5/simple_rnn_5/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
$sequential_5/simple_rnn_5/zeros_like	ZerosLike4sequential_5/simple_rnn_5/simple_rnn_cell_5/Tanh:y:0*
T0*'
_output_shapes
:���������}
2sequential_5/simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������n
,sequential_5/simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �	
sequential_5/simple_rnn_5/whileWhile5sequential_5/simple_rnn_5/while/loop_counter:output:0;sequential_5/simple_rnn_5/while/maximum_iterations:output:0'sequential_5/simple_rnn_5/time:output:02sequential_5/simple_rnn_5/TensorArrayV2_1:handle:0(sequential_5/simple_rnn_5/zeros_like:y:0(sequential_5/simple_rnn_5/zeros:output:02sequential_5/simple_rnn_5/strided_slice_1:output:0Qsequential_5/simple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Ssequential_5/simple_rnn_5/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0Jsequential_5_simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resourceKsequential_5_simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resourceLsequential_5_simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :���������:���������: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *6
body.R,
*sequential_5_simple_rnn_5_while_body_57303*6
cond.R,
*sequential_5_simple_rnn_5_while_cond_57302*M
output_shapes<
:: : : : :���������:���������: : : : : : *
parallel_iterations �
Jsequential_5/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
<sequential_5/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStack(sequential_5/simple_rnn_5/while:output:3Ssequential_5/simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0�
/sequential_5/simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������{
1sequential_5/simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential_5/simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
)sequential_5/simple_rnn_5/strided_slice_3StridedSliceEsequential_5/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:08sequential_5/simple_rnn_5/strided_slice_3/stack:output:0:sequential_5/simple_rnn_5/strided_slice_3/stack_1:output:0:sequential_5/simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_mask
*sequential_5/simple_rnn_5/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
%sequential_5/simple_rnn_5/transpose_2	TransposeEsequential_5/simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:03sequential_5/simple_rnn_5/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :�������������������
-sequential_5/dense_5/Tensordot/ReadVariableOpReadVariableOp6sequential_5_dense_5_tensordot_readvariableop_resource*
_output_shapes

:	*
dtype0m
#sequential_5/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_5/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       }
$sequential_5/dense_5/Tensordot/ShapeShape)sequential_5/simple_rnn_5/transpose_2:y:0*
T0*
_output_shapes
:n
,sequential_5/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_5/dense_5/Tensordot/GatherV2GatherV2-sequential_5/dense_5/Tensordot/Shape:output:0,sequential_5/dense_5/Tensordot/free:output:05sequential_5/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_5/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)sequential_5/dense_5/Tensordot/GatherV2_1GatherV2-sequential_5/dense_5/Tensordot/Shape:output:0,sequential_5/dense_5/Tensordot/axes:output:07sequential_5/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_5/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#sequential_5/dense_5/Tensordot/ProdProd0sequential_5/dense_5/Tensordot/GatherV2:output:0-sequential_5/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_5/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%sequential_5/dense_5/Tensordot/Prod_1Prod2sequential_5/dense_5/Tensordot/GatherV2_1:output:0/sequential_5/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_5/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%sequential_5/dense_5/Tensordot/concatConcatV2,sequential_5/dense_5/Tensordot/free:output:0,sequential_5/dense_5/Tensordot/axes:output:03sequential_5/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$sequential_5/dense_5/Tensordot/stackPack,sequential_5/dense_5/Tensordot/Prod:output:0.sequential_5/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(sequential_5/dense_5/Tensordot/transpose	Transpose)sequential_5/simple_rnn_5/transpose_2:y:0.sequential_5/dense_5/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
&sequential_5/dense_5/Tensordot/ReshapeReshape,sequential_5/dense_5/Tensordot/transpose:y:0-sequential_5/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%sequential_5/dense_5/Tensordot/MatMulMatMul/sequential_5/dense_5/Tensordot/Reshape:output:05sequential_5/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	p
&sequential_5/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	n
,sequential_5/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'sequential_5/dense_5/Tensordot/concat_1ConcatV20sequential_5/dense_5/Tensordot/GatherV2:output:0/sequential_5/dense_5/Tensordot/Const_2:output:05sequential_5/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
sequential_5/dense_5/TensordotReshape/sequential_5/dense_5/Tensordot/MatMul:product:00sequential_5/dense_5/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������	�
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
sequential_5/dense_5/BiasAddBiasAdd'sequential_5/dense_5/Tensordot:output:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������	�
sequential_5/dense_5/SoftmaxSoftmax%sequential_5/dense_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :������������������	�
IdentityIdentity&sequential_5/dense_5/Softmax:softmax:0^NoOp*
T0*4
_output_shapes"
 :������������������	�
NoOpNoOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp.^sequential_5/dense_5/Tensordot/ReadVariableOpC^sequential_5/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpB^sequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpD^sequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp ^sequential_5/simple_rnn_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2^
-sequential_5/dense_5/Tensordot/ReadVariableOp-sequential_5/dense_5/Tensordot/ReadVariableOp2�
Bsequential_5/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpBsequential_5/simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2�
Asequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpAsequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2�
Csequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpCsequential_5/simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp2B
sequential_5/simple_rnn_5/whilesequential_5/simple_rnn_5/while:f b
5
_output_shapes#
!:�������������������
)
_user_specified_namemasking_5_input
�
�
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_57459

inputs

states1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������G
TanhTanhadd:z:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_58114
masking_5_input%
simple_rnn_5_58101:	� 
simple_rnn_5_58103:$
simple_rnn_5_58105:
dense_5_58108:	
dense_5_58110:	
identity��dense_5/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
masking_5/PartitionedCallPartitionedCallmasking_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_masking_5_layer_call_and_return_conditional_losses_57721�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall"masking_5/PartitionedCall:output:0simple_rnn_5_58101simple_rnn_5_58103simple_rnn_5_58105*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_57830�
dense_5/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0dense_5_58108dense_5_58110*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_57869�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	�
NoOpNoOp ^dense_5/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:f b
5
_output_shapes#
!:�������������������
)
_user_specified_namemasking_5_input
�,
�
while_body_58618
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:	�G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_5_matmul_readvariableop_resource:	�E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������w
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���}
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�	
�
,__inference_sequential_5_layer_call_fn_58097
masking_5_input
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:	
	unknown_3:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmasking_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_58069|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
5
_output_shapes#
!:�������������������
)
_user_specified_namemasking_5_input
�
�
B__inference_dense_5_layer_call_and_return_conditional_losses_59048

inputs3
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������	c
SoftmaxSoftmaxBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������	m
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*4
_output_shapes"
 :������������������	z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�|
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_58499

inputsP
=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource:	�L
>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource:Q
?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource:;
)dense_5_tensordot_readvariableop_resource:	5
'dense_5_biasadd_readvariableop_resource:	
identity��dense_5/BiasAdd/ReadVariableOp� dense_5/Tensordot/ReadVariableOp�5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp�4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp�6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp�simple_rnn_5/whileY
masking_5/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    �
masking_5/NotEqualNotEqualinputsmasking_5/NotEqual/y:output:0*
T0*5
_output_shapes#
!:�������������������j
masking_5/Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
masking_5/AnyAnymasking_5/NotEqual:z:0(masking_5/Any/reduction_indices:output:0*4
_output_shapes"
 :������������������*
	keep_dims(|
masking_5/CastCastmasking_5/Any:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :������������������p
masking_5/mulMulinputsmasking_5/Cast:y:0*
T0*5
_output_shapes#
!:��������������������
masking_5/SqueezeSqueezemasking_5/Any:output:0*
T0
*0
_output_shapes
:������������������*
squeeze_dims

���������S
simple_rnn_5/ShapeShapemasking_5/mul:z:0*
T0*
_output_shapes
:j
 simple_rnn_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"simple_rnn_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"simple_rnn_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_sliceStridedSlicesimple_rnn_5/Shape:output:0)simple_rnn_5/strided_slice/stack:output:0+simple_rnn_5/strided_slice/stack_1:output:0+simple_rnn_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
simple_rnn_5/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/zeros/packedPack#simple_rnn_5/strided_slice:output:0$simple_rnn_5/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:]
simple_rnn_5/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    �
simple_rnn_5/zerosFill"simple_rnn_5/zeros/packed:output:0!simple_rnn_5/zeros/Const:output:0*
T0*'
_output_shapes
:���������p
simple_rnn_5/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose	Transposemasking_5/mul:z:0$simple_rnn_5/transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������^
simple_rnn_5/Shape_1Shapesimple_rnn_5/transpose:y:0*
T0*
_output_shapes
:l
"simple_rnn_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_1StridedSlicesimple_rnn_5/Shape_1:output:0+simple_rnn_5/strided_slice_1/stack:output:0-simple_rnn_5/strided_slice_1/stack_1:output:0-simple_rnn_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
simple_rnn_5/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_5/ExpandDims
ExpandDimsmasking_5/Squeeze:output:0$simple_rnn_5/ExpandDims/dim:output:0*
T0
*4
_output_shapes"
 :������������������r
simple_rnn_5/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose_1	Transpose simple_rnn_5/ExpandDims:output:0&simple_rnn_5/transpose_1/perm:output:0*
T0
*4
_output_shapes"
 :������������������s
(simple_rnn_5/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_5/TensorArrayV2TensorListReserve1simple_rnn_5/TensorArrayV2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
Bsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
4simple_rnn_5/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose:y:0Ksimple_rnn_5/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���l
"simple_rnn_5/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$simple_rnn_5/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_2StridedSlicesimple_rnn_5/transpose:y:0+simple_rnn_5/strided_slice_2/stack:output:0-simple_rnn_5/strided_slice_2/stack_1:output:0-simple_rnn_5/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%simple_rnn_5/simple_rnn_cell_5/MatMulMatMul%simple_rnn_5/strided_slice_2:output:0<simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&simple_rnn_5/simple_rnn_cell_5/BiasAddBiasAdd/simple_rnn_5/simple_rnn_cell_5/MatMul:product:0=simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
'simple_rnn_5/simple_rnn_cell_5/MatMul_1MatMulsimple_rnn_5/zeros:output:0>simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"simple_rnn_5/simple_rnn_cell_5/addAddV2/simple_rnn_5/simple_rnn_cell_5/BiasAdd:output:01simple_rnn_5/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:����������
#simple_rnn_5/simple_rnn_cell_5/TanhTanh&simple_rnn_5/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������{
*simple_rnn_5/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
simple_rnn_5/TensorArrayV2_1TensorListReserve3simple_rnn_5/TensorArrayV2_1/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���S
simple_rnn_5/timeConst*
_output_shapes
: *
dtype0*
value	B : u
*simple_rnn_5/TensorArrayV2_2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
simple_rnn_5/TensorArrayV2_2TensorListReserve3simple_rnn_5/TensorArrayV2_2/element_shape:output:0%simple_rnn_5/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:����
Dsimple_rnn_5/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
6simple_rnn_5/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensorsimple_rnn_5/transpose_1:y:0Msimple_rnn_5/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0
*

shape_type0:���
simple_rnn_5/zeros_like	ZerosLike'simple_rnn_5/simple_rnn_cell_5/Tanh:y:0*
T0*'
_output_shapes
:���������p
%simple_rnn_5/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������a
simple_rnn_5/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
simple_rnn_5/whileWhile(simple_rnn_5/while/loop_counter:output:0.simple_rnn_5/while/maximum_iterations:output:0simple_rnn_5/time:output:0%simple_rnn_5/TensorArrayV2_1:handle:0simple_rnn_5/zeros_like:y:0simple_rnn_5/zeros:output:0%simple_rnn_5/strided_slice_1:output:0Dsimple_rnn_5/TensorArrayUnstack/TensorListFromTensor:output_handle:0Fsimple_rnn_5/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0=simple_rnn_5_simple_rnn_cell_5_matmul_readvariableop_resource>simple_rnn_5_simple_rnn_cell_5_biasadd_readvariableop_resource?simple_rnn_5_simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :���������:���������: : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *)
body!R
simple_rnn_5_while_body_58391*)
cond!R
simple_rnn_5_while_cond_58390*M
output_shapes<
:: : : : :���������:���������: : : : : : *
parallel_iterations �
=simple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
/simple_rnn_5/TensorArrayV2Stack/TensorListStackTensorListStacksimple_rnn_5/while:output:3Fsimple_rnn_5/TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0u
"simple_rnn_5/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������n
$simple_rnn_5/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: n
$simple_rnn_5/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
simple_rnn_5/strided_slice_3StridedSlice8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0+simple_rnn_5/strided_slice_3/stack:output:0-simple_rnn_5/strided_slice_3/stack_1:output:0-simple_rnn_5/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maskr
simple_rnn_5/transpose_2/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
simple_rnn_5/transpose_2	Transpose8simple_rnn_5/TensorArrayV2Stack/TensorListStack:tensor:0&simple_rnn_5/transpose_2/perm:output:0*
T0*4
_output_shapes"
 :�������������������
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:	*
dtype0`
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:g
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       c
dense_5/Tensordot/ShapeShapesimple_rnn_5/transpose_2:y:0*
T0*
_output_shapes
:a
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:c
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:a
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: c
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: _
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
dense_5/Tensordot/transpose	Transposesimple_rnn_5/transpose_2:y:0!dense_5/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	c
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	a
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������	�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������	s
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*4
_output_shapes"
 :������������������	u
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*4
_output_shapes"
 :������������������	�
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp6^simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5^simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp7^simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp^simple_rnn_5/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2n
5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp5simple_rnn_5/simple_rnn_cell_5/BiasAdd/ReadVariableOp2l
4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp4simple_rnn_5/simple_rnn_cell_5/MatMul/ReadVariableOp2p
6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp6simple_rnn_5/simple_rnn_cell_5/MatMul_1/ReadVariableOp2(
simple_rnn_5/whilesimple_rnn_5/while:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
E
)__inference_masking_5_layer_call_fn_58521

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_masking_5_layer_call_and_return_conditional_losses_57721n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�������������������:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�,
�
while_body_57764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:	�G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_5_matmul_readvariableop_resource:	�E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������w
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���}
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�,
�
while_body_57954
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:	�G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_5_matmul_readvariableop_resource:	�E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������w
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���}
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
B__inference_dense_5_layer_call_and_return_conditional_losses_57869

inputs3
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:	*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:	Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������	c
SoftmaxSoftmaxBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������	m
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*4
_output_shapes"
 :������������������	z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�

�
1__inference_simple_rnn_cell_5_layer_call_fn_59076

inputs
states_0
unknown:	�
	unknown_0:
	unknown_1:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_57579o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0
�[
�
*sequential_5_simple_rnn_5_while_body_57303P
Lsequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_loop_counterV
Rsequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_maximum_iterations/
+sequential_5_simple_rnn_5_while_placeholder1
-sequential_5_simple_rnn_5_while_placeholder_11
-sequential_5_simple_rnn_5_while_placeholder_21
-sequential_5_simple_rnn_5_while_placeholder_3O
Ksequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_strided_slice_1_0�
�sequential_5_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0�
�sequential_5_simple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor_0e
Rsequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0:	�a
Ssequential_5_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:f
Tsequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:,
(sequential_5_simple_rnn_5_while_identity.
*sequential_5_simple_rnn_5_while_identity_1.
*sequential_5_simple_rnn_5_while_identity_2.
*sequential_5_simple_rnn_5_while_identity_3.
*sequential_5_simple_rnn_5_while_identity_4.
*sequential_5_simple_rnn_5_while_identity_5M
Isequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_strided_slice_1�
�sequential_5_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor�
�sequential_5_simple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensorc
Psequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource:	�_
Qsequential_5_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource:d
Rsequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource:��Hsequential_5/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�Gsequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp�Isequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
Qsequential_5/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
Csequential_5/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem�sequential_5_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0+sequential_5_simple_rnn_5_while_placeholderZsequential_5/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
Ssequential_5/simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
Esequential_5/simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItem�sequential_5_simple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor_0+sequential_5_simple_rnn_5_while_placeholder\sequential_5/simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
Gsequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpRsequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
8sequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMulJsequential_5/simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Osequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Hsequential_5/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpSsequential_5_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
9sequential_5/simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAddBsequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Psequential_5/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
Isequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpTsequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
:sequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul-sequential_5_simple_rnn_5_while_placeholder_3Qsequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5sequential_5/simple_rnn_5/while/simple_rnn_cell_5/addAddV2Bsequential_5/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:0Dsequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:����������
6sequential_5/simple_rnn_5/while/simple_rnn_cell_5/TanhTanh9sequential_5/simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������
.sequential_5/simple_rnn_5/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
$sequential_5/simple_rnn_5/while/TileTileLsequential_5/simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem:item:07sequential_5/simple_rnn_5/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
(sequential_5/simple_rnn_5/while/SelectV2SelectV2-sequential_5/simple_rnn_5/while/Tile:output:0:sequential_5/simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0-sequential_5_simple_rnn_5_while_placeholder_2*
T0*'
_output_shapes
:����������
0sequential_5/simple_rnn_5/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
&sequential_5/simple_rnn_5/while/Tile_1TileLsequential_5/simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem:item:09sequential_5/simple_rnn_5/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
*sequential_5/simple_rnn_5/while/SelectV2_1SelectV2/sequential_5/simple_rnn_5/while/Tile_1:output:0:sequential_5/simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0-sequential_5_simple_rnn_5_while_placeholder_3*
T0*'
_output_shapes
:����������
Dsequential_5/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem-sequential_5_simple_rnn_5_while_placeholder_1+sequential_5_simple_rnn_5_while_placeholder1sequential_5/simple_rnn_5/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:���g
%sequential_5/simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
#sequential_5/simple_rnn_5/while/addAddV2+sequential_5_simple_rnn_5_while_placeholder.sequential_5/simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: i
'sequential_5/simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
%sequential_5/simple_rnn_5/while/add_1AddV2Lsequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_loop_counter0sequential_5/simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: �
(sequential_5/simple_rnn_5/while/IdentityIdentity)sequential_5/simple_rnn_5/while/add_1:z:0%^sequential_5/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
*sequential_5/simple_rnn_5/while/Identity_1IdentityRsequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_maximum_iterations%^sequential_5/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
*sequential_5/simple_rnn_5/while/Identity_2Identity'sequential_5/simple_rnn_5/while/add:z:0%^sequential_5/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
*sequential_5/simple_rnn_5/while/Identity_3IdentityTsequential_5/simple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0%^sequential_5/simple_rnn_5/while/NoOp*
T0*
_output_shapes
: :����
*sequential_5/simple_rnn_5/while/Identity_4Identity1sequential_5/simple_rnn_5/while/SelectV2:output:0%^sequential_5/simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:����������
*sequential_5/simple_rnn_5/while/Identity_5Identity3sequential_5/simple_rnn_5/while/SelectV2_1:output:0%^sequential_5/simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:����������
$sequential_5/simple_rnn_5/while/NoOpNoOpI^sequential_5/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpH^sequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpJ^sequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "]
(sequential_5_simple_rnn_5_while_identity1sequential_5/simple_rnn_5/while/Identity:output:0"a
*sequential_5_simple_rnn_5_while_identity_13sequential_5/simple_rnn_5/while/Identity_1:output:0"a
*sequential_5_simple_rnn_5_while_identity_23sequential_5/simple_rnn_5/while/Identity_2:output:0"a
*sequential_5_simple_rnn_5_while_identity_33sequential_5/simple_rnn_5/while/Identity_3:output:0"a
*sequential_5_simple_rnn_5_while_identity_43sequential_5/simple_rnn_5/while/Identity_4:output:0"a
*sequential_5_simple_rnn_5_while_identity_53sequential_5/simple_rnn_5/while/Identity_5:output:0"�
Isequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_strided_slice_1Ksequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_strided_slice_1_0"�
Qsequential_5_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceSsequential_5_simple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"�
Rsequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceTsequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"�
Psequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceRsequential_5_simple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"�
�sequential_5_simple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor�sequential_5_simple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor_0"�
�sequential_5_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor�sequential_5_simple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_sequential_5_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :���������:���������: : : : : : 2�
Hsequential_5/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpHsequential_5/simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2�
Gsequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpGsequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2�
Isequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpIsequential_5/simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_57763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_57763___redundant_placeholder03
/while_while_cond_57763___redundant_placeholder13
/while_while_cond_57763___redundant_placeholder23
/while_while_cond_57763___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�	
�
,__inference_sequential_5_layer_call_fn_57889
masking_5_input
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:	
	unknown_3:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmasking_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_57876|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
5
_output_shapes#
!:�������������������
)
_user_specified_namemasking_5_input
�=
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58792
inputs_0C
0simple_rnn_cell_5_matmul_readvariableop_resource:	�?
1simple_rnn_cell_5_biasadd_readvariableop_resource:D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������k
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_58726*
condR
while_cond_58725*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
while_cond_58725
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_58725___redundant_placeholder03
/while_while_cond_58725___redundant_placeholder13
/while_while_cond_58725___redundant_placeholder23
/while_while_cond_58725___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�=
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58020

inputsC
0simple_rnn_cell_5_matmul_readvariableop_resource:	�?
1simple_rnn_cell_5_biasadd_readvariableop_resource:D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������k
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_57954*
condR
while_cond_57953*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�

�
1__inference_simple_rnn_cell_5_layer_call_fn_59062

inputs
states_0
unknown:	�
	unknown_0:
	unknown_1:
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_57459o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0
�,
�
while_body_58726
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0K
8while_simple_rnn_cell_5_matmul_readvariableop_resource_0:	�G
9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:L
:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorI
6while_simple_rnn_cell_5_matmul_readvariableop_resource:	�E
7while_simple_rnn_cell_5_biasadd_readvariableop_resource:J
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:��.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�-while/simple_rnn_cell_5/MatMul/ReadVariableOp�/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
-while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp8while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
while/simple_rnn_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:05while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
while/simple_rnn_cell_5/BiasAddBiasAdd(while/simple_rnn_cell_5/MatMul:product:06while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
 while/simple_rnn_cell_5/MatMul_1MatMulwhile_placeholder_27while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
while/simple_rnn_cell_5/addAddV2(while/simple_rnn_cell_5/BiasAdd:output:0*while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������w
while/simple_rnn_cell_5/TanhTanhwhile/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:����������
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder while/simple_rnn_cell_5/Tanh:y:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :���}
while/Identity_4Identity while/simple_rnn_cell_5/Tanh:y:0^while/NoOp*
T0*'
_output_shapes
:����������

while/NoOpNoOp/^while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.^while/simple_rnn_cell_5/MatMul/ReadVariableOp0^while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"t
7while_simple_rnn_cell_5_biasadd_readvariableop_resource9while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"v
8while_simple_rnn_cell_5_matmul_1_readvariableop_resource:while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"r
6while_simple_rnn_cell_5_matmul_readvariableop_resource8while_simple_rnn_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2`
.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp.while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2^
-while/simple_rnn_cell_5/MatMul/ReadVariableOp-while/simple_rnn_cell_5/MatMul/ReadVariableOp2b
/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
'__inference_dense_5_layer_call_fn_59017

inputs
unknown:	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_57869|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
,__inference_sequential_5_layer_call_fn_58152

inputs
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:	
	unknown_3:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_57876|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�4
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_57535

inputs*
simple_rnn_cell_5_57460:	�%
simple_rnn_cell_5_57462:)
simple_rnn_cell_5_57464:
identity��)simple_rnn_cell_5/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
)simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_5_57460simple_rnn_cell_5_57462simple_rnn_cell_5_57464*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_57459n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_5_57460simple_rnn_cell_5_57462simple_rnn_cell_5_57464*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_57472*
condR
while_cond_57471*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������z
NoOpNoOp*^simple_rnn_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2V
)simple_rnn_cell_5/StatefulPartitionedCall)simple_rnn_cell_5/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
,__inference_simple_rnn_5_layer_call_fn_58576

inputs
unknown:	�
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58020|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
*sequential_5_simple_rnn_5_while_cond_57302P
Lsequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_loop_counterV
Rsequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_maximum_iterations/
+sequential_5_simple_rnn_5_while_placeholder1
-sequential_5_simple_rnn_5_while_placeholder_11
-sequential_5_simple_rnn_5_while_placeholder_21
-sequential_5_simple_rnn_5_while_placeholder_3R
Nsequential_5_simple_rnn_5_while_less_sequential_5_simple_rnn_5_strided_slice_1g
csequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_cond_57302___redundant_placeholder0g
csequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_cond_57302___redundant_placeholder1g
csequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_cond_57302___redundant_placeholder2g
csequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_cond_57302___redundant_placeholder3g
csequential_5_simple_rnn_5_while_sequential_5_simple_rnn_5_while_cond_57302___redundant_placeholder4,
(sequential_5_simple_rnn_5_while_identity
�
$sequential_5/simple_rnn_5/while/LessLess+sequential_5_simple_rnn_5_while_placeholderNsequential_5_simple_rnn_5_while_less_sequential_5_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: 
(sequential_5/simple_rnn_5/while/IdentityIdentity(sequential_5/simple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "]
(sequential_5_simple_rnn_5_while_identity1sequential_5/simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :���������:���������: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
�
�
while_cond_57953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_57953___redundant_placeholder03
/while_while_cond_57953___redundant_placeholder13
/while_while_cond_57953___redundant_placeholder23
/while_while_cond_57953___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
� 
�
while_body_57631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_02
while_simple_rnn_cell_5_57653_0:	�-
while_simple_rnn_cell_5_57655_0:1
while_simple_rnn_cell_5_57657_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor0
while_simple_rnn_cell_5_57653:	�+
while_simple_rnn_cell_5_57655:/
while_simple_rnn_cell_5_57657:��/while/simple_rnn_cell_5/StatefulPartitionedCall�
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
/while/simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_simple_rnn_cell_5_57653_0while_simple_rnn_cell_5_57655_0while_simple_rnn_cell_5_57657_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_57579�
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder8while/simple_rnn_cell_5/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:���M
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: �
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :����
while/Identity_4Identity8while/simple_rnn_cell_5/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:���������~

while/NoOpNoOp0^while/simple_rnn_cell_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"@
while_simple_rnn_cell_5_57653while_simple_rnn_cell_5_57653_0"@
while_simple_rnn_cell_5_57655while_simple_rnn_cell_5_57655_0"@
while_simple_rnn_cell_5_57657while_simple_rnn_cell_5_57657_0"0
while_strided_slice_1while_strided_slice_1_0"�
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%: : : : :���������: : : : : 2b
/while/simple_rnn_cell_5/StatefulPartitionedCall/while/simple_rnn_cell_5/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_57630
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_57630___redundant_placeholder03
/while_while_cond_57630___redundant_placeholder13
/while_while_cond_57630___redundant_placeholder23
/while_while_cond_57630___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_59093

inputs
states_01
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������G
TanhTanhadd:z:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0
�=
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58684
inputs_0C
0simple_rnn_cell_5_matmul_readvariableop_resource:	�?
1simple_rnn_cell_5_biasadd_readvariableop_resource:D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������k
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_58618*
condR
while_cond_58617*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�=
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_57830

inputsC
0simple_rnn_cell_5_matmul_readvariableop_resource:	�?
1simple_rnn_cell_5_biasadd_readvariableop_resource:D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������k
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_57764*
condR
while_cond_57763*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�4
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_57694

inputs*
simple_rnn_cell_5_57619:	�%
simple_rnn_cell_5_57621:)
simple_rnn_cell_5_57623:
identity��)simple_rnn_cell_5/StatefulPartitionedCall�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
)simple_rnn_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0simple_rnn_cell_5_57619simple_rnn_cell_5_57621simple_rnn_cell_5_57623*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:���������:���������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_57579n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0simple_rnn_cell_5_57619simple_rnn_cell_5_57621simple_rnn_cell_5_57623*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_57631*
condR
while_cond_57630*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :������������������z
NoOpNoOp*^simple_rnn_cell_5/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2V
)simple_rnn_cell_5/StatefulPartitionedCall)simple_rnn_cell_5/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
,__inference_sequential_5_layer_call_fn_58167

inputs
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:	
	unknown_3:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_58069|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
while_cond_58617
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_58617___redundant_placeholder03
/while_while_cond_58617___redundant_placeholder13
/while_while_cond_58617___redundant_placeholder23
/while_while_cond_58617___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�
�
while_cond_58941
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_58941___redundant_placeholder03
/while_while_cond_58941___redundant_placeholder13
/while_while_cond_58941___redundant_placeholder23
/while_while_cond_58941___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:
�9
�
__inference__traced_save_59205
file_prefix-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_simple_rnn_5_simple_rnn_cell_5_kernel_read_readvariableopN
Jsavev2_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_read_readvariableopB
>savev2_simple_rnn_5_simple_rnn_cell_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableopK
Gsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_m_read_readvariableopU
Qsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_m_read_readvariableopI
Esavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableopK
Gsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_v_read_readvariableopU
Qsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_v_read_readvariableopI
Esavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_simple_rnn_5_simple_rnn_cell_5_kernel_read_readvariableopJsavev2_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_read_readvariableop>savev2_simple_rnn_5_simple_rnn_cell_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableopGsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_m_read_readvariableopQsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_m_read_readvariableopEsavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopGsavev2_adam_simple_rnn_5_simple_rnn_cell_5_kernel_v_read_readvariableopQsavev2_adam_simple_rnn_5_simple_rnn_cell_5_recurrent_kernel_v_read_readvariableopEsavev2_adam_simple_rnn_5_simple_rnn_cell_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :	:	: : : : : :	�::: : : : :	:	:	�:::	:	:	�::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:	: 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�:$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:	: 

_output_shapes
:	:%!

_output_shapes
:	�:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:	: 

_output_shapes
:	:%!

_output_shapes
:	�:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
�
�
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_57579

inputs

states1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0m
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������G
TanhTanhadd:z:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_namestates
�	
`
D__inference_masking_5_layer_call_and_return_conditional_losses_57721

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    q
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*5
_output_shapes#
!:�������������������`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :������������������*
	keep_dims(h
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :������������������\
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:�������������������{
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:������������������*
squeeze_dims

���������]
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�������������������:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_57876

inputs%
simple_rnn_5_57831:	� 
simple_rnn_5_57833:$
simple_rnn_5_57835:
dense_5_57870:	
dense_5_57872:	
identity��dense_5/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
masking_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_masking_5_layer_call_and_return_conditional_losses_57721�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall"masking_5/PartitionedCall:output:0simple_rnn_5_57831simple_rnn_5_57833simple_rnn_5_57835*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_57830�
dense_5/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0dense_5_57870dense_5_57872*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_57869�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	�
NoOpNoOp ^dense_5/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_59110

inputs
states_01
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:2
 matmul_1_readvariableop_resource:
identity

identity_1��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0o
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d
addAddV2BiasAdd:output:0MatMul_1:product:0*
T0*'
_output_shapes
:���������G
TanhTanhadd:z:0*
T0*'
_output_shapes
:���������W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������Y

Identity_1IdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:����������:���������: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:QM
'
_output_shapes
:���������
"
_user_specified_name
states/0
�	
`
D__inference_masking_5_layer_call_and_return_conditional_losses_58532

inputs
identityO

NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    q
NotEqualNotEqualinputsNotEqual/y:output:0*
T0*5
_output_shapes#
!:�������������������`
Any/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������
AnyAnyNotEqual:z:0Any/reduction_indices:output:0*4
_output_shapes"
 :������������������*
	keep_dims(h
CastCastAny:output:0*

DstT0*

SrcT0
*4
_output_shapes"
 :������������������\
mulMulinputsCast:y:0*
T0*5
_output_shapes#
!:�������������������{
SqueezeSqueezeAny:output:0*
T0
*0
_output_shapes
:������������������*
squeeze_dims

���������]
IdentityIdentitymul:z:0*
T0*5
_output_shapes#
!:�������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�������������������:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
simple_rnn_5_while_cond_583906
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_2$
 simple_rnn_5_while_placeholder_38
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58390___redundant_placeholder0M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58390___redundant_placeholder1M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58390___redundant_placeholder2M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58390___redundant_placeholder3M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58390___redundant_placeholder4
simple_rnn_5_while_identity
�
simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :���������:���������: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
�
�
,__inference_simple_rnn_5_layer_call_fn_58543
inputs_0
unknown:	�
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_57535|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:�������������������
"
_user_specified_name
inputs/0
�
�
#__inference_signature_wrapper_58516
masking_5_input
unknown:	�
	unknown_0:
	unknown_1:
	unknown_2:	
	unknown_3:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallmasking_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_57411|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
5
_output_shapes#
!:�������������������
)
_user_specified_namemasking_5_input
�=
�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_59008

inputsC
0simple_rnn_cell_5_matmul_readvariableop_resource:	�?
1simple_rnn_cell_5_biasadd_readvariableop_resource:D
2simple_rnn_cell_5_matmul_1_readvariableop_resource:
identity��(simple_rnn_cell_5/BiasAdd/ReadVariableOp�'simple_rnn_cell_5/MatMul/ReadVariableOp�)simple_rnn_cell_5/MatMul_1/ReadVariableOp�while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:���������c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:�������������������D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
����������
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:����
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:����������*
shrink_axis_mask�
'simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOp0simple_rnn_cell_5_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
simple_rnn_cell_5/MatMulMatMulstrided_slice_2:output:0/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOp1simple_rnn_cell_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
simple_rnn_cell_5/BiasAddBiasAdd"simple_rnn_cell_5/MatMul:product:00simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOp2simple_rnn_cell_5_matmul_1_readvariableop_resource*
_output_shapes

:*
dtype0�
simple_rnn_cell_5/MatMul_1MatMulzeros:output:01simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
simple_rnn_cell_5/addAddV2"simple_rnn_cell_5/BiasAdd:output:0$simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:���������k
simple_rnn_cell_5/TanhTanhsimple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:���F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
���������T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : �
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:00simple_rnn_cell_5_matmul_readvariableop_resource1simple_rnn_cell_5_biasadd_readvariableop_resource2simple_rnn_cell_5_matmul_1_readvariableop_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :���������: : : : : *%
_read_only_resource_inputs
	*
_stateful_parallelism( *
bodyR
while_body_58942*
condR
while_cond_58941*8
output_shapes'
%: : : : :���������: : : : : *
parallel_iterations �
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :������������������*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
���������a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          �
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :������������������k
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :�������������������
NoOpNoOp)^simple_rnn_cell_5/BiasAdd/ReadVariableOp(^simple_rnn_cell_5/MatMul/ReadVariableOp*^simple_rnn_cell_5/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2T
(simple_rnn_cell_5/BiasAdd/ReadVariableOp(simple_rnn_cell_5/BiasAdd/ReadVariableOp2R
'simple_rnn_cell_5/MatMul/ReadVariableOp'simple_rnn_cell_5/MatMul/ReadVariableOp2V
)simple_rnn_cell_5/MatMul_1/ReadVariableOp)simple_rnn_cell_5/MatMul_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
simple_rnn_5_while_cond_582246
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_2$
 simple_rnn_5_while_placeholder_38
4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58224___redundant_placeholder0M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58224___redundant_placeholder1M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58224___redundant_placeholder2M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58224___redundant_placeholder3M
Isimple_rnn_5_while_simple_rnn_5_while_cond_58224___redundant_placeholder4
simple_rnn_5_while_identity
�
simple_rnn_5/while/LessLesssimple_rnn_5_while_placeholder4simple_rnn_5_while_less_simple_rnn_5_strided_slice_1*
T0*
_output_shapes
: e
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/Less:z:0*
T0
*
_output_shapes
: "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : :���������:���������: :::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
::

_output_shapes
:
�
�
,__inference_simple_rnn_5_layer_call_fn_58565

inputs
unknown:	�
	unknown_0:
	unknown_1:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_57830|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:�������������������
 
_user_specified_nameinputs
�
�
G__inference_sequential_5_layer_call_and_return_conditional_losses_58131
masking_5_input%
simple_rnn_5_58118:	� 
simple_rnn_5_58120:$
simple_rnn_5_58122:
dense_5_58125:	
dense_5_58127:	
identity��dense_5/StatefulPartitionedCall�$simple_rnn_5/StatefulPartitionedCall�
masking_5/PartitionedCallPartitionedCallmasking_5_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_masking_5_layer_call_and_return_conditional_losses_57721�
$simple_rnn_5/StatefulPartitionedCallStatefulPartitionedCall"masking_5/PartitionedCall:output:0simple_rnn_5_58118simple_rnn_5_58120simple_rnn_5_58122*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58020�
dense_5/StatefulPartitionedCallStatefulPartitionedCall-simple_rnn_5/StatefulPartitionedCall:output:0dense_5_58125dense_5_58127*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_57869�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������	�
NoOpNoOp ^dense_5/StatefulPartitionedCall%^simple_rnn_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:�������������������: : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2L
$simple_rnn_5/StatefulPartitionedCall$simple_rnn_5/StatefulPartitionedCall:f b
5
_output_shapes#
!:�������������������
)
_user_specified_namemasking_5_input
�K
�
simple_rnn_5_while_body_583916
2simple_rnn_5_while_simple_rnn_5_while_loop_counter<
8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations"
simple_rnn_5_while_placeholder$
 simple_rnn_5_while_placeholder_1$
 simple_rnn_5_while_placeholder_2$
 simple_rnn_5_while_placeholder_35
1simple_rnn_5_while_simple_rnn_5_strided_slice_1_0q
msimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0u
qsimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor_0X
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0:	�T
Fsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0:Y
Gsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0:
simple_rnn_5_while_identity!
simple_rnn_5_while_identity_1!
simple_rnn_5_while_identity_2!
simple_rnn_5_while_identity_3!
simple_rnn_5_while_identity_4!
simple_rnn_5_while_identity_53
/simple_rnn_5_while_simple_rnn_5_strided_slice_1o
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensors
osimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensorV
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource:	�R
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource:W
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource:��;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp�:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp�<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp�
Dsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"�����  �
6simple_rnn_5/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0simple_rnn_5_while_placeholderMsimple_rnn_5/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:����������*
element_dtype0�
Fsimple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
8simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItemTensorListGetItemqsimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor_0simple_rnn_5_while_placeholderOsimple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem/element_shape:output:0*'
_output_shapes
:���������*
element_dtype0
�
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOpReadVariableOpEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	�*
dtype0�
+simple_rnn_5/while/simple_rnn_cell_5/MatMulMatMul=simple_rnn_5/while/TensorArrayV2Read/TensorListGetItem:item:0Bsimple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOpReadVariableOpFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0*
_output_shapes
:*
dtype0�
,simple_rnn_5/while/simple_rnn_cell_5/BiasAddBiasAdd5simple_rnn_5/while/simple_rnn_cell_5/MatMul:product:0Csimple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOpReadVariableOpGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes

:*
dtype0�
-simple_rnn_5/while/simple_rnn_cell_5/MatMul_1MatMul simple_rnn_5_while_placeholder_3Dsimple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(simple_rnn_5/while/simple_rnn_cell_5/addAddV25simple_rnn_5/while/simple_rnn_cell_5/BiasAdd:output:07simple_rnn_5/while/simple_rnn_cell_5/MatMul_1:product:0*
T0*'
_output_shapes
:����������
)simple_rnn_5/while/simple_rnn_cell_5/TanhTanh,simple_rnn_5/while/simple_rnn_cell_5/add:z:0*
T0*'
_output_shapes
:���������r
!simple_rnn_5/while/Tile/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
simple_rnn_5/while/TileTile?simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem:item:0*simple_rnn_5/while/Tile/multiples:output:0*
T0
*'
_output_shapes
:����������
simple_rnn_5/while/SelectV2SelectV2 simple_rnn_5/while/Tile:output:0-simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0 simple_rnn_5_while_placeholder_2*
T0*'
_output_shapes
:���������t
#simple_rnn_5/while/Tile_1/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
simple_rnn_5/while/Tile_1Tile?simple_rnn_5/while/TensorArrayV2Read_1/TensorListGetItem:item:0,simple_rnn_5/while/Tile_1/multiples:output:0*
T0
*'
_output_shapes
:����������
simple_rnn_5/while/SelectV2_1SelectV2"simple_rnn_5/while/Tile_1:output:0-simple_rnn_5/while/simple_rnn_cell_5/Tanh:y:0 simple_rnn_5_while_placeholder_3*
T0*'
_output_shapes
:����������
7simple_rnn_5/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem simple_rnn_5_while_placeholder_1simple_rnn_5_while_placeholder$simple_rnn_5/while/SelectV2:output:0*
_output_shapes
: *
element_dtype0:���Z
simple_rnn_5/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/while/addAddV2simple_rnn_5_while_placeholder!simple_rnn_5/while/add/y:output:0*
T0*
_output_shapes
: \
simple_rnn_5/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
simple_rnn_5/while/add_1AddV22simple_rnn_5_while_simple_rnn_5_while_loop_counter#simple_rnn_5/while/add_1/y:output:0*
T0*
_output_shapes
: �
simple_rnn_5/while/IdentityIdentitysimple_rnn_5/while/add_1:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_1Identity8simple_rnn_5_while_simple_rnn_5_while_maximum_iterations^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_2Identitysimple_rnn_5/while/add:z:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: �
simple_rnn_5/while/Identity_3IdentityGsimple_rnn_5/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^simple_rnn_5/while/NoOp*
T0*
_output_shapes
: :����
simple_rnn_5/while/Identity_4Identity$simple_rnn_5/while/SelectV2:output:0^simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:����������
simple_rnn_5/while/Identity_5Identity&simple_rnn_5/while/SelectV2_1:output:0^simple_rnn_5/while/NoOp*
T0*'
_output_shapes
:����������
simple_rnn_5/while/NoOpNoOp<^simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;^simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp=^simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "C
simple_rnn_5_while_identity$simple_rnn_5/while/Identity:output:0"G
simple_rnn_5_while_identity_1&simple_rnn_5/while/Identity_1:output:0"G
simple_rnn_5_while_identity_2&simple_rnn_5/while/Identity_2:output:0"G
simple_rnn_5_while_identity_3&simple_rnn_5/while/Identity_3:output:0"G
simple_rnn_5_while_identity_4&simple_rnn_5/while/Identity_4:output:0"G
simple_rnn_5_while_identity_5&simple_rnn_5/while/Identity_5:output:0"d
/simple_rnn_5_while_simple_rnn_5_strided_slice_11simple_rnn_5_while_simple_rnn_5_strided_slice_1_0"�
Dsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resourceFsimple_rnn_5_while_simple_rnn_cell_5_biasadd_readvariableop_resource_0"�
Esimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resourceGsimple_rnn_5_while_simple_rnn_cell_5_matmul_1_readvariableop_resource_0"�
Csimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resourceEsimple_rnn_5_while_simple_rnn_cell_5_matmul_readvariableop_resource_0"�
osimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensorqsimple_rnn_5_while_tensorarrayv2read_1_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_1_tensorlistfromtensor_0"�
ksimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensormsimple_rnn_5_while_tensorarrayv2read_tensorlistgetitem_simple_rnn_5_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :���������:���������: : : : : : 2z
;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp;simple_rnn_5/while/simple_rnn_cell_5/BiasAdd/ReadVariableOp2x
:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp:simple_rnn_5/while/simple_rnn_cell_5/MatMul/ReadVariableOp2|
<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp<simple_rnn_5/while/simple_rnn_cell_5/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
while_cond_58833
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_58833___redundant_placeholder03
/while_while_cond_58833___redundant_placeholder13
/while_while_cond_58833___redundant_placeholder23
/while_while_cond_58833___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-: : : : :���������: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:���������:

_output_shapes
: :

_output_shapes
:"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
Y
masking_5_inputF
!serving_default_masking_5_input:0�������������������H
dense_5=
StatefulPartitionedCall:0������������������	tensorflow/serving/predict:�t
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
�
#iter

$beta_1

%beta_2
	&decay
'learning_ratemXmY(mZ)m[*m\v]v^(v_)v`*va"
	optimizer
C
(0
)1
*2
3
4"
trackable_list_wrapper
C
(0
)1
*2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
�
+non_trainable_variables

,layers
-metrics
.layer_regularization_losses
/layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_sequential_5_layer_call_fn_57889
,__inference_sequential_5_layer_call_fn_58152
,__inference_sequential_5_layer_call_fn_58167
,__inference_sequential_5_layer_call_fn_58097�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_sequential_5_layer_call_and_return_conditional_losses_58333
G__inference_sequential_5_layer_call_and_return_conditional_losses_58499
G__inference_sequential_5_layer_call_and_return_conditional_losses_58114
G__inference_sequential_5_layer_call_and_return_conditional_losses_58131�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_57411masking_5_input"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
0serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
)__inference_masking_5_layer_call_fn_58521�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_masking_5_layer_call_and_return_conditional_losses_58532�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�

(kernel
)recurrent_kernel
*bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:_random_generator
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
�

=states
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_simple_rnn_5_layer_call_fn_58543
,__inference_simple_rnn_5_layer_call_fn_58554
,__inference_simple_rnn_5_layer_call_fn_58565
,__inference_simple_rnn_5_layer_call_fn_58576�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58684
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58792
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58900
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_59008�
���
FullArgSpecB
args:�7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults�

 
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 :	2dense_5/kernel
:	2dense_5/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cnon_trainable_variables

Dlayers
Emetrics
Flayer_regularization_losses
Glayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
�2�
'__inference_dense_5_layer_call_fn_59017�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_5_layer_call_and_return_conditional_losses_59048�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
8:6	�2%simple_rnn_5/simple_rnn_cell_5/kernel
A:?2/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel
1:/2#simple_rnn_5/simple_rnn_cell_5/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
#__inference_signature_wrapper_58516masking_5_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
6	variables
7trainable_variables
8regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
�2�
1__inference_simple_rnn_cell_5_layer_call_fn_59062
1__inference_simple_rnn_cell_5_layer_call_fn_59076�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_59093
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_59110�
���
FullArgSpec3
args+�(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ototal
	Pcount
Q	variables
R	keras_api"
_tf_keras_metric
^
	Stotal
	Tcount
U
_fn_kwargs
V	variables
W	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
O0
P1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
S0
T1"
trackable_list_wrapper
-
V	variables"
_generic_user_object
%:#	2Adam/dense_5/kernel/m
:	2Adam/dense_5/bias/m
=:;	�2,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/m
F:D26Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/m
6:42*Adam/simple_rnn_5/simple_rnn_cell_5/bias/m
%:#	2Adam/dense_5/kernel/v
:	2Adam/dense_5/bias/v
=:;	�2,Adam/simple_rnn_5/simple_rnn_cell_5/kernel/v
F:D26Adam/simple_rnn_5/simple_rnn_cell_5/recurrent_kernel/v
6:42*Adam/simple_rnn_5/simple_rnn_cell_5/bias/v�
 __inference__wrapped_model_57411�(*)F�C
<�9
7�4
masking_5_input�������������������
� ">�;
9
dense_5.�+
dense_5������������������	�
B__inference_dense_5_layer_call_and_return_conditional_losses_59048v<�9
2�/
-�*
inputs������������������
� "2�/
(�%
0������������������	
� �
'__inference_dense_5_layer_call_fn_59017i<�9
2�/
-�*
inputs������������������
� "%�"������������������	�
D__inference_masking_5_layer_call_and_return_conditional_losses_58532t=�:
3�0
.�+
inputs�������������������
� "3�0
)�&
0�������������������
� �
)__inference_masking_5_layer_call_fn_58521g=�:
3�0
.�+
inputs�������������������
� "&�#��������������������
G__inference_sequential_5_layer_call_and_return_conditional_losses_58114�(*)N�K
D�A
7�4
masking_5_input�������������������
p 

 
� "2�/
(�%
0������������������	
� �
G__inference_sequential_5_layer_call_and_return_conditional_losses_58131�(*)N�K
D�A
7�4
masking_5_input�������������������
p

 
� "2�/
(�%
0������������������	
� �
G__inference_sequential_5_layer_call_and_return_conditional_losses_58333�(*)E�B
;�8
.�+
inputs�������������������
p 

 
� "2�/
(�%
0������������������	
� �
G__inference_sequential_5_layer_call_and_return_conditional_losses_58499�(*)E�B
;�8
.�+
inputs�������������������
p

 
� "2�/
(�%
0������������������	
� �
,__inference_sequential_5_layer_call_fn_57889~(*)N�K
D�A
7�4
masking_5_input�������������������
p 

 
� "%�"������������������	�
,__inference_sequential_5_layer_call_fn_58097~(*)N�K
D�A
7�4
masking_5_input�������������������
p

 
� "%�"������������������	�
,__inference_sequential_5_layer_call_fn_58152u(*)E�B
;�8
.�+
inputs�������������������
p 

 
� "%�"������������������	�
,__inference_sequential_5_layer_call_fn_58167u(*)E�B
;�8
.�+
inputs�������������������
p

 
� "%�"������������������	�
#__inference_signature_wrapper_58516�(*)Y�V
� 
O�L
J
masking_5_input7�4
masking_5_input�������������������">�;
9
dense_5.�+
dense_5������������������	�
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58684�(*)P�M
F�C
5�2
0�-
inputs/0�������������������

 
p 

 
� "2�/
(�%
0������������������
� �
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58792�(*)P�M
F�C
5�2
0�-
inputs/0�������������������

 
p

 
� "2�/
(�%
0������������������
� �
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_58900�(*)I�F
?�<
.�+
inputs�������������������

 
p 

 
� "2�/
(�%
0������������������
� �
G__inference_simple_rnn_5_layer_call_and_return_conditional_losses_59008�(*)I�F
?�<
.�+
inputs�������������������

 
p

 
� "2�/
(�%
0������������������
� �
,__inference_simple_rnn_5_layer_call_fn_58543~(*)P�M
F�C
5�2
0�-
inputs/0�������������������

 
p 

 
� "%�"�������������������
,__inference_simple_rnn_5_layer_call_fn_58554~(*)P�M
F�C
5�2
0�-
inputs/0�������������������

 
p

 
� "%�"�������������������
,__inference_simple_rnn_5_layer_call_fn_58565w(*)I�F
?�<
.�+
inputs�������������������

 
p 

 
� "%�"�������������������
,__inference_simple_rnn_5_layer_call_fn_58576w(*)I�F
?�<
.�+
inputs�������������������

 
p

 
� "%�"�������������������
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_59093�(*)]�Z
S�P
!�
inputs����������
'�$
"�
states/0���������
p 
� "R�O
H�E
�
0/0���������
$�!
�
0/1/0���������
� �
L__inference_simple_rnn_cell_5_layer_call_and_return_conditional_losses_59110�(*)]�Z
S�P
!�
inputs����������
'�$
"�
states/0���������
p
� "R�O
H�E
�
0/0���������
$�!
�
0/1/0���������
� �
1__inference_simple_rnn_cell_5_layer_call_fn_59062�(*)]�Z
S�P
!�
inputs����������
'�$
"�
states/0���������
p 
� "D�A
�
0���������
"�
�
1/0����������
1__inference_simple_rnn_cell_5_layer_call_fn_59076�(*)]�Z
S�P
!�
inputs����������
'�$
"�
states/0���������
p
� "D�A
�
0���������
"�
�
1/0���������