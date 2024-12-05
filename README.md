# JaPyGuri_dummy

모델을 시작하기 전에 백엔드로부터 이미지를 받아 ./input_image에 저장이 됐음을 가정한다.
모델의 실행 순서는 다음과 같다.

1. output_image에 존재하는 모든 파일을 지운다. 이전에 생성됐던 피팅 파일이 이 시점에서 사라진다.
2. 입력이미지를 모델이 입력해 손톱 영역을 segement한다
3. 2번의 영역을 기반으로 하여 사전에 준비된 네일아트 이미지를 마스킹한다.
4. 최종 이미지파일을 ./output_image파일에 저장한다. 백엔드 서버는 해당 폴더로 부터 파일을 획득한다.
5. ./input_iamge의 모든 파일을 지운다. 이로써 모델이 시작하면 항상 입력 이미지가 1개만 있게 제약한다.

