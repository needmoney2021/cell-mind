# CellMind – **분산 미니-GPT** 프로젝트 가이드 (2025-04-25 rev)

당연하게도 챗지피티와 커서가 다 만들었습니다.
**아직 테스트 못 해봤어요. 단말기가 없어서....ㅎㅎ**

## 0. 목차
1. [사전 준비](#prereq)  
2. [데이터·토크나이저 준비](#data)  
3. [`shared/config.json`] 설정법  
4. [메인 단말 (Orchestrator)](#orch)  
5. [워커 단말](#worker)  
6. [추론 / 체크포인트 동기화](#infer-sync)  
7. [기타 팁](#tips)  

---

<a name="prereq"></a>
## 1. 사전 준비

```bash
# 1) 저장소 클론
git clone {repos}
cd cellmind

# 2) 가상환경
python -m venv venv
source venv/bin/activate        # Win: venv\Scripts\activate

# 3) 의존성
pip install -r requirements.txt
```

> **모든 단말(메인·워커)에서 동일하게** 수행해야 합니다.

---

<a name="data"></a>
## 2. 데이터 & 토크나이저

| 단계 | 파일/디렉터리                                                                                                                | 설명                                     |
|------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| ① | `data/raw/*.txt`                                                                                                       | 원본 로 데이터                               |
| ② | **토크나이저 학습**<br>`python -m tokenizer`                                                                                  | → `{path}/tokenizer.model`             |
| ③ | **JSONL 변환**<br>각 줄 `{"text": "..."} `                                                                                 | → `data/for-model-training/train.jsonl` |
| ④ | 학습 시 **`TextDataset`** 가 이 JSONL을 읽어 자동으로 `<\|start\|> … <\|endoftext\|>` 토큰을 붙이고, `collate_fn` 으로 배치 패딩(-100 mask) 처리 |

---

## 3. `shared/config.json` 예시

```jsonc
{
  "distributed": {
    "backend": "gloo",                   // CPU-only → gloo, GPU → nccl
    "init_method": "tcp://192.168.0.100:23456",
    "world_size": 3,                     // 1(메인) + 2(워커)
    "worker_ips": [
      "192.168.0.101",
      "192.168.0.102"
    ],
    "usernames": {
      "192.168.0.101": "laptop_user",
      "192.168.0.102": "termux_user"
    }
  },

  "model": {
    "d_model":     512,
    "nhead":       8,
    "num_layers":  6,
    "dropout":     0.1,
    "checkpoint":  "checkpoints/latest.pt"   // 최초 학습 시 비워둬도 됨
  },

  "tokenizer": {
    "model_path":  "tokenizer/tokenizer.model",
    "vocab_size":  8000
  },

  "training": {
    "batch_size":  32,
    "epochs":      3,
    "lr":          3e-4,
    "dataset": {
      "path":      "data/for-model-training/train.jsonl",
      "seq_len":   256
    }
  }
}
```

---

<a name="orch"></a>
## 4. 메인 단말 (Orchestrator)

### 4-1. SSH 준비 (최초 1회)
```bash
ssh-keygen -t rsa -b 4096 -N ""      # 키 없으면 생성
ssh-copy-id laptop_user@192.168.0.101
ssh-copy-id termux_user@192.168.0.102
```

### 4-2. 학습 시작
```bash
python orchestrator/manager.py --mode train
```
* `manager.py` 가 각 워커에 **환경변수(MASTER_ADDR · RANK 등)** 를 포함한
  `python -m worker.processor --mode train` 명령을 SSH로 전송 → 학습 자동 시작.

### 4-3. 실시간 로그
메인 단말 콘솔에 rank 0 로그가 출력되고, 워커 콘솔에는 각자 rank 로그가 뜹니다.

---

<a name="worker"></a>
## 5. 워커 단말

| 항목 | Linux laptop | Android (예: Termux) |
|------|-------------|-----------------------|
| 레포 클론·venv·pip install | 메인과 동일 | ```pkg install python git openssh``` 후 동일 |
| SSH 서버 | 보통 이미 `sshd` 실행 중 | ```sshd``` 실행 |
| **실행 필요 없음** | Orchestrator가 알아서 프로세스를 띄움 | Orchestrator가 알아서 프로세스를 띄움 |

---

<a name="infer-sync"></a>
## 6. 추론 & 체크포인트 동기화

| 모드 | 명령 |
|------|------|
| **추론** | `python orchestrator/manager.py --mode inference --prompt "안녕 GPT!"` |
| **파일로 추론** | `python orchestrator/manager.py --mode inference --prompt-file prompt.txt` |
| **체크포인트만 강제 sync** | `python orchestrator/manager.py --mode sync` |

---

<a name="tips"></a>
## 7. 기타 팁

* **GPU** 단말은 PyTorch + CUDA 설치 후 `distributed.backend` 을 `nccl` 로.  
* 학습 로그·그래프를 보고 싶으면 `DistributedTrainer`에 **TensorBoard** 코드(네 줄) 끼워 넣으면 됨.  
* 데이터셋 크기가 수 GiB 이상이면 `TextDataset` 을 **메모리-매핑(mmap)** 방식으로 바꾸는 것을 추천.  
