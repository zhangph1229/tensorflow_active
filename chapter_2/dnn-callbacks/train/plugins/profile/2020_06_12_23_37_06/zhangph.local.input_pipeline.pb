	j�t�@j�t�@!j�t�@	�l�U���?�l�U���?!�l�U���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j�t�@/�$��?A�z�G�@YV-��?*	     �K@2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapZd;�O��?!袋.��D@)Zd;�O��?1袋.��D@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�� �rh�?!袋.��>@)�� �rh�?1袋.��>@:Preprocessing2F
Iterator::Model���Q��?!E]t�E;@)����Mb�?1]t�E-@:Preprocessing2S
Iterator::Model::ParallelMapy�&1�|?!t�E]t)@)y�&1�|?1t�E]t)@:Preprocessing2R
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.9% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	/�$��?/�$��?!/�$��?      ��!       "      ��!       *      ��!       2	�z�G�@�z�G�@!�z�G�@:      ��!       B      ��!       J	V-��?V-��?!V-��?R      ��!       Z	V-��?V-��?!V-��?JCPU_ONLY