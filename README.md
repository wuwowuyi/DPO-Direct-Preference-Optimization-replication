
My implementation of the [DPO paper](https://arxiv.org/abs/2305.18290), implemented from scratch using Pytorch.


GPT2 small model is OpenAI GPT-2 model with 124M parameters, hosted by Hugging face.

## Sentiment
This is a stylistic continuation task, and model is encouraged to generate a "positive and happy" continuation of a query which is sampled from a training dataset, like an excerpt from the BookCorpus dataset.

### GPT2 small


| query                                                                                                                                                                                                                                                                                            |  reference policy                                                          | DPO trained policy                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------- |------------------------------------------------------------------------------------------------|
| He did not like things which reminded him that he had once been not only outside the progressive element but even outside the College. He did not always like Curry either. His pleasure in being with him was not that sort of pleasure.<br>"Yes," said Curry. "Denniston was your chief rival. | He wanted you to win. He was a man who could win a lot of games.           | I look forward to working with her." Denniston smiled. "I look forward to working with you."   |
| Corolan moved to the next chair down, giving me his. Wylend floated my water glass across the table. Water was the only thing Wyatt would allow me to drink. I'd already had my broth.                                                                                                           | Gaffney was a bit restless. I was sure he'd found a way to kill him        | I had a good wine and a good hot tea and a good coffee. I was ready for the serious challenge. |
| She took too many hits in the head playing football and soccer and they ruined her mind. She has fighter's dementia, similar to what the Quarry brothers had. Do you want to be that way, dear?"<br>Sean blinked. He hadn't known why Ruby was always shaking. He was silent for a moment.       |  "I could tell you," he said. "You might always know, or feel what she did.| "Well, thank God, she was fine. She was maybe even a good soccer player.                       |

