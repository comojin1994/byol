import requests
from configparser import ConfigParser

from pytorch_lightning.callbacks import ModelCheckpoint

def get_callbacks(args):
    ckpt_callback = ModelCheckpoint(every_n_epochs=100,
                                    dirpath=f'{args.CKPT_PATH}/{args.current_time}',
                                    filename='{epoch}',
                                    save_top_k=-1)

    return [ckpt_callback]

        
def post_message(channel, text):
    config = ConfigParser()
    config.read('key.ini')
    token = config['Key']['token'][1:-1]
    
    try:
        response = requests.post("https://slack.com/api/chat.postMessage",
            headers={"Authorization": "Bearer "+token},
            data={"channel": channel,"text": text}
        )
        print(response)
    except Exception as e:
        print(e)
        
        