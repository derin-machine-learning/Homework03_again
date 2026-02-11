import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import string

from abc import abstractmethod
from difflib import SequenceMatcher
try:
  from gradio_client import Client, handle_file
except:
  print("no gradio")
from warnings import simplefilter

class FFTEncoder:
  def __init__(self, slide_freq=3000):
    self.slide_freq = slide_freq

  def encode(self, ins):
    outs = np.array(ins)
    hertz_per_sample = 44100 / len(ins)
    start_slide_sample = int(self.slide_freq / hertz_per_sample)

    outs[0] = 0 + 0j
    for s in range(1, len(ins) // 2):
      s2 = len(ins) - s
      if s < start_slide_sample:
        outs[s] = 0 + 0j
      else:
        outs[s] = ins[s - start_slide_sample]
      outs[s2] = np.conjugate(outs[s])

    return outs

  def decode(self, ins):
    outs = np.array(ins)
    hertz_per_sample = 44100 / len(ins)
    start_slide_sample = int(self.slide_freq / hertz_per_sample)

    outs[0] = 0 + 0j
    for s in range(1, len(ins) // 2):
      s2 = len(ins) - s
      if s < len(ins) // 2 - start_slide_sample:
        outs[s] = ins[s + start_slide_sample]
      else:
        outs[s] = 0 + 0j
      outs[s2] = np.conjugate(outs[s])

      return outs


class SpeechToText:
  client = None

  @staticmethod
  def transcribe_file(filepath):
    result = SpeechToText.client.predict(
		  audio_in=handle_file(filepath),
		  api_name="/transcribe"
    )
    return result.translate(str.maketrans('', '', string.punctuation))

  @staticmethod
  def transcribe(samples_or_filepath):
    simplefilter(action="ignore")
    if SpeechToText.client is None:
      SpeechToText.client = Client("visualizedata/5020-STT-Gradio")
    if type(samples_or_filepath) == str:
      return SpeechToText.transcribe_file(samples_or_filepath)
    else:
      fname = "./tmp.wav"
      sf.write(fname, samples_or_filepath, 44100, subtype="PCM_16")
      return SpeechToText.transcribe_file(fname)


class Homework03Utils():
  INSTRUMENTS = ["clarinet", "guitar", "piano"]
  L2I = {v:i for i,v in enumerate(INSTRUMENTS)}
  MMAP = {
    "01": "the next sound is backwards",
    "02": "the next sound got squished move the smallest valued samples back to the center",
    "03": "the next message got zipped read every 13th sample",
    "04": "this message is not sound just read every 19th sample into a square image to see",
  }

  @staticmethod
  def compare_transcription(samples, text):
    text_h = SpeechToText.transcribe(samples)
    score = SequenceMatcher(None, text_h, text).ratio()
    if score < 0.6:
      return f"ERROR: Decoded message \"{text_h}\" is really different from expected result."
    elif score < 0.85:
      return f"ERROR: Decoded message \"{text_h}\" is similar to \"{text}\", but still different."
    else:
      return f"Message decoded correctly:\n\t\"{text}\" 🎉🎉🎉"

  @staticmethod
  def peek_secret(test_label):
    if test_label not in Homework03Utils.MMAP:
      raise Exception(f"Can't find: {test_label}. Check test name.")
    return Homework03Utils.MMAP[test_label]

  @staticmethod
  def transcribe(samples):
    return SpeechToText.transcribe(samples)

  @staticmethod
  def test_transcription(test_label, samples=None):
    test_text = Homework03Utils.peek_secret(test_label)
    test_result = Homework03Utils.compare_transcription(samples, test_text)
    print(f"{test_label}: {test_result}")

  @staticmethod
  def PRIME_SEED(i):
    return [2081, 2087, 2089][i]

  @staticmethod
  def plot_labels_vals(vls, title):
    l2v = {}
    for v,l in vls:
      label = l.split(".")[0].split("-")[0]
      l2v[label] = l2v.get(label, []) + [v]

    xs = l2v.values()
    ys = [len(v) * [i] for i,v in enumerate(xs)]

    plt.figure(figsize=(8, 2))
    plt.scatter(xs, ys)
    plt.yticks(range(0, len(l2v.keys())), list(l2v.keys()))
    plt.title(title)
    plt.show()

  @staticmethod
  def classification_accuracy(labels_and_filenames):
    preds = { i:[] for i in Homework03Utils.INSTRUMENTS }
    acc = {}

    for label,fname in labels_and_filenames:
      correct_label = function(fname)
      preds[correct_label].append(label)

    for label, label_preds in preds.items():
      correct = [1 for pred in label_preds if pred == label]
      pct = 0 if len(label_preds) == 0 else sum(correct) / len(label_preds)
      acc[label] = round(pct, 5)

    acc["overall"] = round(sum(acc.values()) / len(acc.values()), 5)
    return acc

def function(x):
  if "-" in x:
    return x.split("-")[0]
  else:
    x_int = int(x.split(".")[0])
    i_idx = [x_int % Homework03Utils.PRIME_SEED(i) == 0 for i in Homework03Utils.L2I.values()].index(True)
    return Homework03Utils.INSTRUMENTS[i_idx]
