$	???Q?a@?x,??i@??????!?c> P?q@	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?c> P?q@1?? @@IRF\ ?o@"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??????1??c?~??I??iܛ??*	?????ye@2U
Iterator::Model::ParallelMapV2p_?Q??!i?v#?=@)p_?Q??1i?v#?=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea??+e??!?E"7??D@)䃞ͪϥ?1??F???8@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??Ɯ?!????1[0@)??Ɯ?1????1[0@:Preprocessing2F
Iterator::Model?????ױ?!Q??ID@)46<???1l)o??M%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??A?f??!s??P?T(@)r??????1?w????$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????9#??!??/ ??M@)??ǘ????1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-C??6j?!$E? V???)-C??6j?1$E? V???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?lV}????![??;UFE@)a2U0*?c?1?3?? Z??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?88.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIw??%.V@Q=G??ю&@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	^h??H!0@[Y+??6@??c?~??!?? @@*	!       2	!       :$	?70?Q?_@???,(+f@??iܛ??!RF\ ?o@B	!       J	!       R	!       Z	!       b	!       JGPUb qw??%.V@y=G??ю&@