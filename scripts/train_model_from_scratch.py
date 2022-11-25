from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.display_data import DisplayData
from parlai.scripts.train_model import TrainModel
from parlai.scripts.interactive import Interactive
import os

os.system("rm -rf from_scratch_model")
os.system("mkdir -p from_scratch_model")

@register_teacher("colab_chatbot")
class MyTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # opt is the command line arguments.
        
        # What is this shared thing?
        # We make many copies of a teacher, one-per-batchsize. Shared lets us store 
        
        # We just need to set the "datafile".  This is boilerplate, but differs in many teachers.
        # The "datafile" is the filename where we will load the data from. In this case, we'll set it to
        # the fold name (train/valid/test) + ".txt"
        opt['datafile'] = opt['datatype'].split(':')[0] + ".txt"
        super().__init__(opt, shared)
    
    def setup_data(self, datafile='train.txt'):
        # setup_data should yield tuples of ((text, label), new_episode)
        # That is ((str, str), bool)
        print(f" ~~ Loading from file {datafile} ~~ ")
        f = open(datafile)
        full_text = f.read()
        full_text_split = full_text.split("\n")
        for text in full_text_split:
            text_list = text.split("\t")
            if len(text_list) > 1:
                sp1 = text_list[0].replace("text:", "")
                sp2 = text_list[1].replace("labels:", "")
                yield ({"text": sp1, "labels": sp2}, True)
        
        

TrainModel.main(
    # we MUST provide a filename
    model_file='from_scratch_model/model',
    # train on empathetic dialogues
    task='colab_chatbot',
    # limit training time to 2 minutes, and a batchsize of 16
    max_train_time=60*2,
    batchsize=16,
    
    # we specify the model type as seq2seq
    model='seq2seq',
    # some hyperparamter choices. We'll use attention. We could use pretrained
    # embeddings too, with embedding_type='fasttext', but they take a long
    # time to download.
    attention='dot',
    # tie the word embeddings of the encoder/decoder/softmax.
    lookuptable='all',
    # truncate text and labels at 64 tokens, for memory and time savings
    truncate=64,
)

wanna_chat = input("Would you like to chat? Y/[N]")
if wanna_chat.lower() == 'y':    
    # call it with particular args
    Interactive.main(
        # the model_file is a filename path pointing to a particular model dump.
        # Model files that begin with "zoo:" are special files distributed by the ParlAI team.
        # They'll be automatically downloaded when you ask to use them.
        model_file='from_scratch_model/model'
    )