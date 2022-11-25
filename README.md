## Installation
Just navigate to this folder and run docker compose up or create a virtual environement.

We use virtualenv to create a virtual environment you can install it with 
```
pip install virtualenv
```

Create a virtual environment and install the requirements.txt

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## To run project
Run the model server with 
```
python3 run.py --config-path ./config.yml --port 8080
```

## To change the blender model
Add a blender model download script in dwn_model, there is a model example already there

Change the model docker RUN script Dockerfile

```
RUN python3 dwn_model/blender_90M.py
to
RUN python3 dwn_model/OTHERMODEL.py
```

Change the model name and setting in config.yml under

```
models:
    blender_90M:
    ...
```

## To change custom model

Copy your model files in customworld
example is how the mental_bert_swmh_improved is placed there

Then edit the worlds.py file with the required lib and write the inference script in the world.py
you can call the inference in the def parley function it and print it there. example can be seen in the world.py under following function
```
class MessengerBotChatTaskWorld(World):
    def parley(self):
    ....
```
now re build the docker with docker compose build