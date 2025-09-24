#!/usr/bin/env python3
"""
경사하강법 하이퍼파라미터 최적화 실시간 모니터링 도구

기능:
1. 실시간 로그 파싱 및 진행상황 표시
2. 성능 지표 그래프 생성
3. 최적 파라미터 추적
4. 예상 완료 시간 계산
"""

import json
import time
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from collections import deque

class OptimizationMonitor:
    """최적화 과정 실시간 모니터링"""

    def __init__(self):
        self.log_file = "/root/workspace/data/raw/gradient_optimization.log"
        self.progress_file = "/root/workspace/data/raw/optimization_progress.json"
        self.results_file = "/root/workspace/data/raw/gradient_optimization_results.json"
        self.pid_file = "/root/workspace/data/raw/optimization_pid.txt"

        self.iterations = []
        self.alphas = []
        self.scores = []
        self.best_scores = []
        self.start_time = None

    def is_running(self):
        """최적화 프로세스가 실행 중인지 확인"""
        try:
            if not os.path.exists(self.pid_file):
                return False

            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # 프로세스 존재 여부 확인
            return os.path.exists(f"/proc/{pid}")
        except:
            return False

    def parse_log_line(self, line):
        """로그 라인 파싱"""
        # 반복 정보 파싱: "반복   10: α=1.234567, 손실=0.123456, 최적손실=0.123456, 시간=1.23s"
        pattern = r'반복\s+(\d+):\s+α=([\d.]+),\s+손실=([\d.-]+),\s+최적손실=([\d.-]+),\s+시간=([\d.]+)s'
        match = re.search(pattern, line)

        if match:
            iteration = int(match.group(1))
            alpha = float(match.group(2))
            score = float(match.group(3))
            best_score = float(match.group(4))
            elapsed_time = float(match.group(5))

            return {
                'iteration': iteration,
                'alpha': alpha,
                'score': score,
                'best_score': best_score,
                'elapsed_time': elapsed_time,
                'timestamp': datetime.now()
            }
        return None

    def update_progress_plot(self):
        """진행상황 플롯 업데이트"""
        if len(self.iterations) < 2:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 손실 함수 (음의 R²)
        ax1.plot(self.iterations, self.scores, 'b-', alpha=0.7, label='현재 손실')
        ax1.plot(self.iterations, self.best_scores, 'r-', linewidth=2, label='최적 손실')
        ax1.set_xlabel('반복 횟수')
        ax1.set_ylabel('손실 (-R²)')
        ax1.set_title('손실 함수 수렴')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Alpha 파라미터 변화
        ax2.plot(self.iterations, self.alphas, 'g-', linewidth=2)
        ax2.set_xlabel('반복 횟수')
        ax2.set_ylabel('Alpha 값')
        ax2.set_title('Ridge Alpha 파라미터 최적화')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 3. R² 점수 (양수로 변환)
        r2_scores = [-score for score in self.scores]
        best_r2_scores = [-score for score in self.best_scores]

        ax3.plot(self.iterations, r2_scores, 'b-', alpha=0.7, label='현재 R²')
        ax3.plot(self.iterations, best_r2_scores, 'r-', linewidth=2, label='최적 R²')
        ax3.set_xlabel('반복 횟수')
        ax3.set_ylabel('R² 점수')
        ax3.set_title('모델 성능 (R² 점수)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 수렴 분석
        if len(self.best_scores) > 10:
            # 최근 10개 값의 표준편차 (수렴 지표)
            window_size = 10
            convergence = []
            for i in range(window_size, len(self.best_scores)):
                window = self.best_scores[i-window_size:i]
                convergence.append(np.std(window))

            ax4.plot(self.iterations[window_size:], convergence, 'm-', linewidth=2)
            ax4.set_xlabel('반복 횟수')
            ax4.set_ylabel('최근 10회 표준편차')
            ax4.set_title('수렴 분석')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 플롯 저장
        plot_path = '/root/workspace/data/raw/optimization_progress.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    def estimate_completion_time(self):
        """완료 예상 시간 계산"""
        if len(self.iterations) < 5:
            return None

        # 최근 개선이 없던 횟수 추정
        recent_improvements = []
        best_so_far = float('inf')

        for score in self.best_scores:
            if score < best_so_far:
                best_so_far = score
                recent_improvements.append(0)
            else:
                if recent_improvements:
                    recent_improvements[-1] += 1
                else:
                    recent_improvements.append(1)

        if not recent_improvements:
            return None

        current_patience = recent_improvements[-1] if recent_improvements else 0
        max_patience = 30  # 설정값

        remaining_patience = max_patience - current_patience

        if remaining_patience <= 0:
            return "곧 완료 예정"

        # 평균 반복 시간 계산
        if len(self.iterations) > 1:
            time_per_iteration = (datetime.now() - self.start_time).total_seconds() / len(self.iterations)
            estimated_remaining_time = remaining_patience * time_per_iteration * 10  # 10회마다 로깅

            return f"약 {int(estimated_remaining_time/60)}분 {int(estimated_remaining_time%60)}초 남음"

        return None

    def print_status(self):
        """현재 상태 출력"""
        os.system('clear')

        print("🎯 경사하강법 하이퍼파라미터 최적화 모니터링")
        print("=" * 60)

        # 프로세스 상태
        if self.is_running():
            print("📊 상태: 🟢 실행 중")
        else:
            print("📊 상태: 🔴 중지됨")

        if self.iterations:
            print(f"🔄 현재 반복: {self.iterations[-1]}")
            print(f"📈 현재 Alpha: {self.alphas[-1]:.6f}")
            print(f"📊 현재 손실: {self.scores[-1]:.6f}")
            print(f"🏆 최적 손실: {self.best_scores[-1]:.6f}")
            print(f"🎖️  최적 R²: {-self.best_scores[-1]:.6f}")

            # 기존 성능과 비교
            baseline_r2 = 0.3113
            current_best_r2 = -self.best_scores[-1]
            improvement = current_best_r2 - baseline_r2
            improvement_pct = (improvement / baseline_r2) * 100

            print(f"📈 기존 성능: R² = {baseline_r2:.4f}")
            print(f"📊 성능 향상: {improvement:+.4f} ({improvement_pct:+.2f}%)")

            # 완료 예상 시간
            eta = self.estimate_completion_time()
            if eta:
                print(f"⏰ 완료 예상: {eta}")

        print("\n" + "=" * 60)
        print("💡 명령어:")
        print("   Ctrl+C: 모니터링 중단")
        print("   로그 확인: tail -f /root/workspace/data/raw/gradient_optimization.log")
        print("=" * 60)

    def monitor(self):
        """실시간 모니터링 실행"""
        print("🚀 최적화 모니터링 시작...")

        last_position = 0
        self.start_time = datetime.now()

        try:
            while True:
                # 로그 파일 읽기
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()

                        # 새로운 라인 파싱
                        for line in new_lines:
                            parsed = self.parse_log_line(line)
                            if parsed:
                                self.iterations.append(parsed['iteration'])
                                self.alphas.append(parsed['alpha'])
                                self.scores.append(parsed['score'])
                                self.best_scores.append(parsed['best_score'])

                # 상태 출력
                self.print_status()

                # 플롯 업데이트 (매 30초마다)
                if len(self.iterations) > 0 and len(self.iterations) % 3 == 0:
                    try:
                        plot_path = self.update_progress_plot()
                        print(f"📊 플롯 업데이트: {plot_path}")
                    except Exception as e:
                        print(f"⚠️ 플롯 업데이트 실패: {e}")

                # 최적화 완료 확인
                if os.path.exists(self.results_file):
                    print("\n🎉 최적화 완료! 결과를 확인하세요.")
                    with open(self.results_file, 'r') as f:
                        results = json.load(f)

                    print(f"🏆 최적 Alpha: {results['best_hyperparameters']['alpha']:.6f}")
                    print(f"🎯 최종 R²: {results['best_performance']['r2_score']:.6f}")
                    break

                if not self.is_running():
                    print("\n⚠️ 최적화 프로세스가 중지되었습니다.")
                    break

                time.sleep(10)  # 10초마다 업데이트

        except KeyboardInterrupt:
            print("\n🛑 모니터링을 중단합니다.")

def main():
    monitor = OptimizationMonitor()
    monitor.monitor()

if __name__ == "__main__":
    main()