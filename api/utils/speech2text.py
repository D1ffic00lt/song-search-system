from transformers import pipeline


class Speech2TextModel(object):
    TASK = "automatic-speech-recognition"
    def __init__(self, model_name: str, *, verbose: bool = True):
        self.model = self.load(model_name, verbose=verbose)

    @staticmethod
    def load(model_name: str, *, verbose: bool = True):
        if not verbose:
            return pipeline(Speech2TextModel.TASK, model=model_name)
        print("Loading pipeline...")
        model = pipeline(Speech2TextModel.TASK, model=model_name)
        print("Pipeline loaded.")
        return model

    def predict(self, audio_path: str) -> str:
        transcription = self.model(audio_path)
        return transcription['text']
