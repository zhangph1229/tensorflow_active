"�#
VHostIDLE"IDLE(1     ��@9     �W@A     ��@I     �W@aUK����?iUK����?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1      e@9      e@A      e@I      e@a:_���?iv6�'&��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     �c@9     �c@A     �c@I     �c@a*��Wù?i����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1     �\@9     �\@A     �\@I     �\@aqUZ;���?iffffff�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      T@9      T@A      T@I      T@aKz���?i����?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1     �O@9     �O@A     �O@I     �O@aݺ���?i�ܫ�P�?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      C@9      C@A      C@I      C@aB����Ș?iKz���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1     �B@9     �B@A     �B@I     �B@ab��y�!�?i,JM���?�Unknown
�	HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      B@9      B@A      B@I      B@a�C!��z�?iHݺ��?�Unknown
�
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      A@9      A@A      A@I      A@a��t� -�?iF���"E�?�Unknown
^HostGatherV2"GatherV2(1      >@9      >@A      >@I      >@aA����?i��ܫ��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      0@9      0@A      0@I      0@a<��߄?i���(5�?�Unknown
�HostDataset"3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat(1      ,@9      ,@A      *@I      *@a����<��?i����x�?�Unknown
�HostDataset"-Iterator::Model::ParallelMap::Zip[0]::FlatMap(1      (@9      (@A      (@I      (@aZ,��N?io�@���?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1      &@9      &@A      &@I      &@a�R����|?i������?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      $@9      $@A      $@I      $@aKz��z?i��vT-%�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@a�C!��zw?i1�j8#T�?�Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a<���t?i��`X�}�?�Unknown
qHostDataset"Iterator::Model::ParallelMap(1      @9      @A      @I      @a�4o�-Cr?ifW�g��?�Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aZ,��No?il�OL���?�Unknown
dHostDataset"Iterator::Model(1      *@9      *@A      @I      @aZ,��No?iƾG���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @aZ,��No?i �?|S �?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aZ,��No?iz8��?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @aZ,��No?i�C0��>�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aKz��j?i�)�Y�?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aKz��j?ij8#Ts�?�Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aKz��j?i��(5��?�Unknown
�HostDataset"MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a<���d?i�z8��?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a<���d?i-CH��?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @aZ,��N_?iZY����?�Unknown
fHostGreaterEqual"GreaterEqual(1      @9      @A      @I      @aZ,��N_?i�o
�A��?�Unknown
� HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aZ,��N_?i��,���?�Unknown
}!HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      @9      @A      @I      @aZ,��N_?i�x���?�Unknown
�"HostDataset"?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor(1      �?9      �?A      �?I      �?a<���D?i�M<���?�Unknown
j#HostReadVariableOp"ReadVariableOp(1      �?9      �?A      �?I      �?a<���D?i�������?�Unknown*�#
oHost_FusedMatMul"sequential/dense/Relu(1      e@9      e@A      e@I      e@a���#��?i���#��?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1     �c@9     �c@A     �c@I     �c@a�b:��,�?i(������?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1     �\@9     �\@A     �\@I     �\@a      �?i(������?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1      T@9      T@A      T@I      T@aLg1��t�?i~��G�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1     �O@9     �O@A     �O@I     �O@a��k(��?i�Gp�}�?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1      C@9      C@A      C@I      C@aUUUUUU�?i1��t��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1     �B@9     �B@A     �B@I     �B@aZLg1�Ť?i��Gp�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      B@9      B@A      B@I      B@a^Cy�5�?i-����b�?�Unknown
�	Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      A@9      A@A      A@I      A@ag1��t�?iCy�5��?�Unknown
^
HostGatherV2"GatherV2(1      >@9      >@A      >@I      >@ay�5�נ?i�k(���?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      0@9      0@A      0@I      0@ap�}��?i�YLg1�?�Unknown
�HostDataset"3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat(1      ,@9      ,@A      *@I      *@a�YLg1�?ib:��,��?�Unknown
�HostDataset"-Iterator::Model::ParallelMap::Zip[0]::FlatMap(1      (@9      (@A      (@I      (@a(�����?i�}��?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1      &@9      &@A      &@I      &@a:��,���?iLg1��t�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1      $@9      $@A      $@I      $@aLg1��t�?i�,�����?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      "@9      "@A      "@I      "@a^Cy�5�?i��Gp�?�Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @ap�}��?it�YLg�?�Unknown
qHostDataset"Iterator::Model::ParallelMap(1      @9      @A      @I      @a��Gp?ib:��,��?�Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a(�����z?i�}���?�Unknown
dHostDataset"Iterator::Model(1      *@9      *@A      @I      @a(�����z?i�}��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a(�����z?i|��G�?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a(�����z?i�Gp�}�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1      @9      @A      @I      @a(�����z?i8��,���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aLg1��tv?i�#����?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aLg1��tv?i�P^Cy�?�Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @aLg1��tv?i����b:�?�Unknown
�HostDataset"MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @ap�}�q?i�5��P^�?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @ap�}�q?i#���>��?�Unknown
tHostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a(�����j?i�YLg1��?�Unknown
fHostGreaterEqual"GreaterEqual(1      @9      @A      @I      @a(�����j?i���#��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a(�����j?i0��t��?�Unknown
} HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1      @9      @A      @I      @a(�����j?i�>����?�Unknown
�!HostDataset"?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor(1      �?9      �?A      �?I      �?ap�}�Q?io�}��?�Unknown
j"HostReadVariableOp"ReadVariableOp(1      �?9      �?A      �?I      �?ap�}�Q?i�������?�Unknown