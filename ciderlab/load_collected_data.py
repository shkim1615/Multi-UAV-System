import torch

# 출력 옵션을 크게 설정 (주의: 너무 큰 텐서는 파일 크기가 커질 수 있음)
torch.set_printoptions(threshold=10000000000)

# 저장된 파일 불러오기
collected_data = torch.load("ciderlab/collected_data/collected_data_90.pt")

print("Keys:", collected_data.keys())
print()

# 'agents' 그룹의 'episode_reward' 데이터의 첫 배치(예: 첫 에피소드)를 가져오기
reward_sample = collected_data.get(("agents", "episode_reward"))[0]

# 텐서를 문자열로 변환 (전체 내용)
output_text = str(reward_sample)

# 텍스트 파일로 저장
with open("ciderlab/agent_reward_details.txt", "w") as f:
    f.write(output_text)

print("에이전트 Reward 정보가 'agent_reward_details.txt' 파일에 저장되었습니다.")
