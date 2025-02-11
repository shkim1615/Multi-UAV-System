
from my_vmas.scenarios.cooperative_exploration import Scenario
from my_vmas.interactive_rendering import render_interactively

test = Scenario()
print("시나리오 클래스 생성")
test.make_world(batch_dim=1, device="cpu")
print("월드 생성 완료")