# Bert2Bert Summarization
Abstractive summarization using [Bert2Bert](https://arxiv.org/pdf/1907.12461.pdf) framework
![](https://user-images.githubusercontent.com/38183241/101169654-9d6fdd80-3680-11eb-96ae-965d22b05fa8.png)
<br><br>

## Dataset
- I used dataset from Dacon's summarization competition.
- You can find dataset I uesd [here](https://dacon.io/competitions/official/235673/leaderboard/).
- Put the `train.jsonl` and `abstractive_test_v2.jsonl` into `dataset` folder.
<br><br>

## Modeling
- [KoBERT](https://github.com/SKTBrain/KoBERT) (from SKT) is used for Bert2Bert framework.
- I used [KoBERT-transformers](https://github.com/monologg/KoBERT-Transformers) and [Pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) to reduce modeling codes. 
<br><br>

## Training
- Run `train.py` to train bert2bert model.
- I used 4 * V100 GPU to train model for 25k step.
<br><br>

## Evaluation
- Run `test.py` to evaluate bert2bert model.
- You can make submission file using `make_submission.py {CKPT_STEP}`.
- You can also find codes that calculate ROUGE score [here](https://dacon.io/competitions/official/235673/talkboard/401911?page=1&dtype=recent&ptype=pub).
<br><br>

#### Quantitive Evalutation
> |Metric|Score|
> |-------|----|
> |ROUGE-1|44.8|
> |ROUGE-2|25.8|
> |ROUGE-L|35.2|

#### Qualitive Evaluation
> Original Text : ▲ 석문간척지 임차법인협의회가 한국농어촌공사 당진지사 앞에 공공비축벼 320t을 쌓아두고 시위를 벌이고 있다. 석문간척지 임차법인협의회(이하 간척지협의회)가 농림축산식품부의 부당한 간척지 임대료 책정에 반발하며 지난달 30일 한국농어촌공사 당진지사 앞에 공공비축벼 320t을 쌓고 시위를 벌였다. 43개 영농조합법인이 소속된 간척지협의회는 이번 벼 야적 시위를 통해 현재 1kg당 2100원으로 책정된 임대료를 현재 쌀 판매가격인 1300원대로 인하할 것을 요구하고 있다. 이들은 지난 12월 7일 농림축산식품부에 탄원서를 제출했지만 “임대료 인하는 올해 이후에나 가능하다”고 통보받은 상황이다. 게다가 임차법인들의 계약기간이 올해 만료되기 때문에 임대료를 인하해도 지난 2년 동안의 손실 보상은 받을 수 없는 상황이다. 이에 간척지협의회는 계약기간을 2년 연장하고, 연장된 기간 동안 인하된 임대료를 적용해 지난 2년 간의 손실에 대해 보상할 것을 제안했다. 더불어 요구사항이 받아들여지지 않을 경우 벼 야적시위를 시작한 날짜인 지난해 12월 30일자로 임대료를 벼로 납부하겠다는 입장이다. 김재용 봉치영농조합법인 조합원은 “현재 한국농어촌공사의 답변을 기다리고 있다”며 “상황을 지켜본 뒤 추가적인 야적 시위 여부를 결정할 계획”이라고 말했다. 이어 “법을 만들고 집행할 때 현실성 있게 만들어야 한다”며 “농민이 정부를 믿을 수 있도록 조속히 해결책을 마련해주길 바란다”고 덧붙였다.
---
> Summarized : 석문간척지 임차법인협의회가 지난달 30일 한국농어촌공사 당진지사 앞에서 농림축산식품부의 부당한 간척지 임대료 책정에 반발하며 벼 야적장 320t을 쌓아놓고 인근 영농인들에게 공공비축쌀 임대료를 받을 수 있도록 인하해 달라고 요구했다.

<br>

## Licence
    Copyright 2020 Hyunwoong Ko.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
    http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
