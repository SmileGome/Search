# MathMatch : 수식 인식 및 검색 서비스
<p>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3766AB?style=flat-square&logo=Python&logoColor=white"/></a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/></a>
  <a href="https://www.pytorchlightning.ai/">
    <img src="https://img.shields.io/badge/PyTorch Lightning-792EE5?style=flat-square&logo=PyTorch Lightning&logoColor=white"/></a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white"/></a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=FastAPI&logoColor=white"/></a>
</p>

- 입력하기 번거로운 수식을 OCR기능으로 입력하고, 수식의 개념을 검색할 수 있는 서비스 개발
- OCR 기능과 검색 기능을 하나로 엮은 데모 완성을 통해 서비스화 가능성 확보
- 스마일 게이트 희망스튜디오 퓨처랩 AI 부문 2기 지원 프로젝트

## Latex OCR 구조도
<img src="https://user-images.githubusercontent.com/68782183/218077351-efeed6de-1834-4ac4-b570-bd20b20d01ee.png" width=600 heigth=300>

## Latex Search 구조도
<img src="https://user-images.githubusercontent.com/68782183/218077368-a620bdfb-d478-4537-98c0-7961141a2215.png" width=600 heigth=300>

***

## Latex Search
본 repo는 수식 검색 모델을 위한 것입니다.

### DPR 모델
DPR 모델은 BERT인코더로 Query와 Passage를 인코딩하고 이들의 dot product값이 positive sample일 경우 커지도록 학습하는 것입니다.

저희는 위키피디아의 수식 데이터를 크롤링하였고, 한 도큐먼트 내에 등장하는 수식들을 positive sample로 처리하였습니다.

### 코드 설명

`dataset.py`

수식 데이터를 query, positive sample, negative sample 형태로 만드는 모듈

`train.py`

DPR모델을 학습하는 모듈

`tokenizer.py`

수식 맞춤형 토크나이저를 만드는 모듈

`inference.py` 

입력된 수식을 임베딩화하고 이를 리턴하는 모듈

`embedding.py`

elasticDB에 저장할 임베딩을 만드는 모듈 



# Team GOME
| [강지우](https://github.com/jiwoo0212) | [곽지윤](https://github.com/kwakjeeyoon) | [김윤혜](https://github.com/yoonene) | 
| :-: | :-: | :-: | 
| <img src="https://user-images.githubusercontent.com/68782183/146319428-ea9b3554-53d3-46e3-aa41-a0a07660fbab.png" width=100 height=100> | <img src="https://user-images.githubusercontent.com/68782183/146319494-b789dff2-a2c4-49a1-a3f0-29eb5e3f3cf7.png" width=100 height=100> | <img src="https://avatars.githubusercontent.com/u/56261032?v=4" width=100 height=100> |


