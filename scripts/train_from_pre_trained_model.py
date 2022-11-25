import os
from parlai.scripts.train_model import TrainModel
from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.interactive import Interactive


os.system("rm -rf from_pretrained")
os.system("mkdir -p from_pretrained")

@register_teacher("emphatic_teacher")
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
                yield ((sp1, sp2), True)
        
TrainModel.main(
    # similar to before
    task='emphatic_teacher', 
    model="image_seq2seq",
    model_file='from_pretrained/model',
    
    # initialize with a pretrained model
    init_model='zoo:dodecadialogue/empathetic_dialogues_ft/model',
    dict_lower=True, 
    dict_tokenizer='bpe',

    dict_file='zoo:dodecadialogue/empathetic_dialogues_ft/model.dict',
    eps=0.5,
    betas=0.9,
    warmup_updates=2000,
    gradient_clip=0.1,
    fp16=False,
    esz=512,
    ffn_size=2048,
    n_heads=16,
    n_layers=8,
    variant='xlm',
    activation='gelu',
    n_positions=512,
    text_truncate=512,
    label_truncate=128,
    lr=7e-6,
    lr_scheduler='reduceonplateau',
    # optimizer='adamax',
    dropout=0.1,
    validation_every_n_secs=3600,
    validation_metric='ppl',
    validation_metric_mode='min',
    validation_patience=10,
    embeddings_scale=True,
    learn_positional_embeddings=True
)

wanna_chat = input("Would you like to chat? Y/[N]")
if wanna_chat.lower() == 'y':  
    # call it with particular args
    Interactive.main(
        # the model_file is a filename path pointing to a particular model dump.
        # Model files that begin with "zoo:" are special files distributed by the ParlAI team.
        # They'll be automatically downloaded when you ask to use them.
        # model_file='from_pretrained/model'
        model_file='from_pretrained/model'
    )