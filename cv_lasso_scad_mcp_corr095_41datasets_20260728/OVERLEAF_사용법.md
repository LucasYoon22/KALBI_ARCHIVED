# Overleaf 사용법

## 1. 업로드

Overleaf → **New Project** → **Upload Project** → `V14_진단_overleaf.zip` 선택

또는 빈 프로젝트를 만들고 `main.tex` 하나만 올려도 됩니다.

## 2. 컴파일러 설정 (필수)

**Menu → Settings → Compiler → `XeLaTeX`**

pdfLaTeX 로는 한글이 컴파일되지 않습니다.

## 3. 폰트

Overleaf 에서는 `kotex` 패키지가 자동으로 잡힙니다. 별도 설정이 필요 없습니다.

폰트를 바꾸시려면 preamble 의 kotex 블록 아래에 추가하십시오.

```latex
\setmainhangulfont{Noto Serif CJK KR}
```

## 4. 구조

| 절 | 내용 |
|---|---|
| 1 | 진단 대상과 전제, 분석의 제약 |
| 2 | 1순위 — 명제 1(항등식), 명제 2(집중), 차원 효과, 실측, 수정 방향 |
| 3 | 2순위 — penalty sparsity 능력 |
| 4 | 3순위 — shape 하한, 난이도 이중봉 |
| 5 | 확정 / 미확정 |
| 6 | 다음 단계 |
| 부록 A~C | 순위 변천사, 기각된 가설, 데이터 출처 |

## 5. 정리 환경

`\begin{lemma}`, `\begin{proposition}`, `\begin{corollary}` 로 보조정리·명제·귀결을 씁니다.
번호는 자동으로 매겨집니다.

## 6. 로컬 컴파일

```bash
xelatex main.tex
xelatex main.tex      # 상호참조 때문에 두 번
```
