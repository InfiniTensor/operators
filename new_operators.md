## Clip

### 基本属性

| 描述                              | 裁剪， 把输入张量中的元素限制在给定的范围之内         |
| --------------------------------- | -----------------------------------------------  |
| 是否支持原地（in-place）计算      | 支持原地                                            |
| 是否需要额外工作空间（workspace） | 不需要                                              |

公式为：

Y=min(max(X,min_val),max_val)

### 接口定义

#### 创建算子描述

```C
infiniopStatus_t infiniopCreateClipDescriptor(infiniopHandle_t handle,
infiniopClipDescriptor_t *desc_ptr,infiniopTensorDescriptor_t dst,
infiniopTensorDescriptor_t src, float max, float min);
```

**参数说明**

| handle   | 硬件控柄                                           |
| -------- | -------------------------------------------------- |
| desc_ptr | 算子描述符的地址                                    |
| dst      | 输出张量描述。形状与src相同。类型与src相同            |
| src      | 输入张量描述。形状不定，类型可以为fp16               |
| max      | float类型，所取值的上界                             |
| min      | float 类型，所取值的下界                            |

**返回值**

| STATUS_SUCCESS              | 成功                             |
| --------------------------- | ------------------------         |
| STATUS_BAD_PARAM            | 参数张量不统一                    |
| STATUS_BAD_TENSOR_DTYPE     | 输入输出张量类型不被支持或两者不一致|
| STATUS_BAD_TENSOR_SHAPE     | 输入输出张量形状不符合要求         |
| STATUS_BAD_TENSOR_STRIDES   | 张量步长不符合要求                |

#### 计算

```C
infiniopStatus_t infiniopClip(infiniopClipDescriptor_t desc,
void *dst, void const *src, void *stream);
```

#### 删除算子描述

```C
infiniopStatus_t infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc);
```





## Where

### 基本属性

| 描述                              | 条件选择， 根据一个条件张量从两个输入张量中选择元素，构建输出张量 |
| --------------------------------- | ----------------------------------------------- |
| 是否支持原地（in-place）计算      | 支持原地                                            |
| 是否需要额外工作空间（workspace） | 不需要                                              |

公式为：
          
output[i]=（condition[i] ? X[i] : Y[i]）
          
### 接口定义

#### 创建算子描述

```C
infiniopStatus_t infiniopCreateWhereDescriptor(infiniopHandle_t handle,
infiniopWhereDescriptor_t *desc_ptr, infiniopTensorDescriptor_t dst,
infiniopTensorDescriptor_t x,infiniopTensorDescriptor_t y,
infiniopTensorDescriptor_t condition);
```

**参数说明**

| handle         | 硬件控柄                                               |
| -------------- | --------------------------------------------------    |
| desc_ptr       | 算子描述符的地址                                        |
| dst            | 输出张量描述。形状与x，y和condition相同。类型与x和y相同   |
| x              | 输入张量描述。形状不定，类型可以为fp16                   |
| y              | 输入张量描述。形状与x相同，类型与x相同                    |
| condition      | 输入条件张量描述。形状与x和y相同，类型为uint8_t           |
                            

**返回值**

| STATUS_SUCCESS              | 成功                                                               |
| --------------------------- | --------------------------------------------------------          |
| STATUS_BAD_PARAM            | 参数张量不统一                                                     |
| STATUS_BAD_TENSOR_DTYPE     | (1)输入输出张量类型不被支持或两者不一致  (2)条件张量类型不支持         |
| STATUS_BAD_TENSOR_SHAPE     | (1)输入输出张量形状不符合要求  (2)条件张量形状不与输入输出张量形状一致  |
| STATUS_BAD_TENSOR_STRIDES   | 张量步长不符合要求                                                  |

#### 计算

```C
infiniopStatus_t infiniopWhere(infiniopWhereDescriptor_t desc,
void *dst,void const *x,void const *y,void const *condition,void *stream);
```

#### 删除算子描述

```C
infiniopStatus_t infiniopDestroyWhereDescriptor(infiniopWhereDescriptor_t desc);
```


## Gather

### 基本属性

| 描述                              | 索引收集， 根据指定的索引从输入数据中收集元素，并将这些元素组织成一个新的张量输出 |
| --------------------------------- | ------------------------------------------------------------------------  |
| 是否支持原地（in-place）计算      | 不支持原地                                                                   |
| 是否需要额外工作空间（workspace） | 不需要                                                                       |

公式为：
          
在指定的axis上 output = data[index]
          
### 接口定义

#### 创建算子描述

```C
infiniopStatus_t infiniopCreateGatherDescriptor(infiniopHandle_t handle,infiniopGatherDescriptor_t *desc_ptr,infiniopTensorDescriptor_t dst,
infiniopTensorDescriptor_t data,infiniopTensorDescriptor_t indices,
int axis);
```

**参数说明**

| handle         | 硬件控柄                                                                                 |
| -------------- | --------------------------------------------------                                      |
| desc_ptr       | 算子描述符的地址                                                                         |
| dst            | 输出张量描述。形状是data_shape[:axis]+indices_shape+data_shape[axis+1:]相同 类型与data相同 |
| data           | 输入张量描述。形状不定，类型可以为fp16                                                     |
| indices        | 索引张量描述。形状不定，类型为一切int32或int64                                             |
| axis           | int类型，表示指定操作的轴， 其值不能超过data的秩                                            |
                 
                            

**返回值**

| STATUS_SUCCESS              | 成功                                                           |
| --------------------------- | ------------------------------------------------------         |
| STATUS_BAD_PARAM            | 参数张量不统一     (1)可能是axis取值不合理                       |
| STATUS_BAD_TENSOR_DTYPE     | (1)输入输出张量类型不被支持或两者不一致  (2)索引值张量类型不被支持  |
| STATUS_BAD_TENSOR_SHAPE     | (1)输入输出张量形状不符合要求  (2)索引张量形状不符合要求           |
| STATUS_BAD_TENSOR_STRIDES   | 张量步长不符合要求                                              |

#### 计算

```C
infiniopStatus_t infiniopGather(infiniopGatherDescriptor_t desc,
void *dst,  void const *data, void const *indices, void *stream);
```

#### 删除算子描述

```C
infiniopStatus_t infiniopDestroyGatherDescriptor(infiniopGatherDescriptor_t desc);
```


## Reduce

### 基本属性

| 描述                               | 归约， 沿指定的一个或多个维度应用一个二元操作（求最值，求加法），输入张量缩减为形状较小的输出张量 |
| --------------------------------- | ----------------------------------------------------------------------------------------  |
| 是否支持原地（in-place）计算        | 不支持原地                                                                                 |
| 是否需要额外工作空间（workspace）   | 需要                                                                                       |

公式为：
          
在指定的axis上 计算(Max,Min,Mean)
          
### 接口定义

#### 创建算子描述

```C
infiniopStatus_t infiniopCreateReduce(Max,Min,Mean)Descriptor(infiniopHandle_t handle,
infiniopReduce(Max,Min,Mean)Descriptor_t *desc_ptr,infiniopTensorDescriptor_t dst,
infiniopTensorDescriptor_t src,int* axis,const int num_axis, int const keepdims);
```

**参数说明**

| handle         | 硬件控柄                                                                                 |
| -------------- | ---------------------------------------------------------------------------             |
| desc_ptr       | 算子描述符的地址                                                                         |
| dst            | 输出张量描述。形状取决于keepdims，保留时rank(dst)=rank(src)，被规约的轴大小为1；不保留时rank(dst) = rank(src)-rank(axis), 相当于去除冗余维度， 类型与src相同 |
| src            | 输入张量描述。形状不定，类型可以为fp16                                                     |
| axis           | int类型的数组，形状小于src的秩，指定需要reduce的轴，支持多个轴同时规约                       |
| num_axis       | axis数组的形状大小。大小小于src的秩，类型为int                                             |
| keepdims       | 确定是否保留形状。为正整数时保留，类型为int                                                 |
                 
                            

**返回值**

| STATUS_SUCCESS              | 成功                             |
| --------------------------- | ------------------------                         |
| STATUS_BAD_PARAM            | 参数张量不统一     (1)可能是axis取值不合理 (2)可能是选择的规约模式不支持          |
| STATUS_BAD_TENSOR_DTYPE     | 输入输出张量类型不被支持或两者不一致|
| STATUS_BAD_TENSOR_SHAPE     | 输入输出张量形状不符合要求         |
| STATUS_BAD_TENSOR_STRIDES   | 张量步长不符合要求                |


#### 计算工作空间

```C
infiniopStatus_t infiniopGetReduce(Max,Min,Mean)WorkspaceSize(infiniopReduce(Max,Min,Mean)Descriptor_t desc, uint64_t *size); 
```

#### 计算

```C
infiniopStatus_t infiniopReduce(Max,Min,Mean)(infiniopReduce(Max,Min,Mean)Descriptor_t desc, 
void *workspace, uint64_t workspace_size, void *dst, void const *src, void *stream);

```

#### 删除算子描述

```C
infiniopStatus_t infiniopDestroyReduce(Max,Min,Mean)Descriptor(infiniopReduce(Max,Min,Mean)Descriptor_t desc);
```