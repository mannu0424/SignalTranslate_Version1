?$	???Q?a@?x,??i@??????!?c> P?q@	!       "\
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
	!       	!       "$	^h??H!0@[Y+??6@??c?~??!?? @@*	!       2	!       :$	?70?Q?_@???,(+f@??iܛ??!RF\ ?o@B	!       J	!       R	!       Z	!       b	!       JGPUb qw??%.V@y=G??ю&@?"?
?gradient_tape/sequential/lstm_2/while/sequential/lstm_2/while_grad/body/_1778/gradient_tape/sequential/lstm_2/while/gradients/AddN_6AddN >ݪU???! >ݪU???"?
?gradient_tape/sequential/lstm_1/while/sequential/lstm_1/while_grad/body/_2143/gradient_tape/sequential/lstm_1/while/gradients/AddN_6AddN,?j?܃?!?)?
?ߓ?"f
Ksequential/lstm_2/while/body/_707/sequential/lstm_2/while/lstm_cell_2/splitSplit?*W??
??!??oVe??"?
?gradient_tape/sequential/lstm_2/while/sequential/lstm_2/while_grad/body/_1778/gradient_tape/sequential/lstm_2/while/gradients/sequential/lstm_2/while/lstm_cell_2/split_grad/concatConcatV2X?J?}?!???g??"^
Csequential/lstm/while/body/_1/sequential/lstm/while/lstm_cell/splitSplit`?Q??y?!???????"f
Ksequential/lstm_1/while/body/_354/sequential/lstm_1/while/lstm_cell_1/splitSplit?n? ?y?!{E'#?٨?"g
Lsequential/lstm_3/while/body/_1060/sequential/lstm_3/while/lstm_cell_3/splitSplitz2R
y?!*?mm?	??"?
?gradient_tape/sequential/lstm_1/while/sequential/lstm_1/while_grad/body/_2143/gradient_tape/sequential/lstm_1/while/gradients/sequential/lstm_1/while/lstm_cell_1/split_grad/concatConcatV2a???5x?!V?&?	??"?
?gradient_tape/sequential/lstm/while/sequential/lstm/while_grad/body/_2508/gradient_tape/sequential/lstm/while/gradients/sequential/lstm/while/lstm_cell/split_grad/concatConcatV2#?N8?w?!]?I?)???"?
?gradient_tape/sequential/lstm_3/while/sequential/lstm_3/while_grad/body/_1413/gradient_tape/sequential/lstm_3/while/gradients/sequential/lstm_3/while/lstm_cell_3/split_grad/concatConcatV2??{?Jw?!?????s??Q      Y@Y?셤
??a???V??X@qt?ci?]W@y#?Y?_r?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?88.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?93.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 