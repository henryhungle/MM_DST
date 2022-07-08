from datasets.dst_datasets import DstDataset 


class DatasetsFactory(object):

    @staticmethod
    def get_training_dst_dataset(model_name: str, video_ft_dir: str, video_labels_dir: str, 
            dial_dir: str, resnext_dir, config, vocab=None, state_vocab=None, res_vocab=None):
        
        return DstDataset(video_ft_dir, video_labels_dir,dial_dir, resnext_dir, vocab, state_vocab, res_vocab, config)

    
    @staticmethod
    def get_inference_dst_dataset(model_name, video_ft_dir, video_labels_dir, 
        dial_dir, resnext_dir, config, vocab, state_vocab, res_vocab): 

        return DstDataset(video_ft_dir, video_labels_dir, dial_dir, resnext_dir, vocab, state_vocab, res_vocab, config)

