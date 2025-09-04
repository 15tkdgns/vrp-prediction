/**
 * Event Bus for Decoupled Communication
 * 컴포넌트 간 느슨한 결합을 위한 이벤트 버스
 */

export class EventBus {
  constructor() {
    this.events = new Map();
    this.onceEvents = new Set();
    this.maxListeners = 10; // 메모리 누수 방지
  }

  /**
   * 이벤트 리스너 등록
   */
  on(eventName, callback, context = null) {
    if (!this.events.has(eventName)) {
      this.events.set(eventName, []);
    }

    const listeners = this.events.get(eventName);
    
    // 최대 리스너 체크
    if (listeners.length >= this.maxListeners) {
      console.warn(`⚠️ EventBus: Too many listeners for "${eventName}" (${listeners.length})`);
    }

    const listener = { callback, context };
    listeners.push(listener);

    // 구독 해제 함수 반환
    return () => this.off(eventName, callback);
  }

  /**
   * 일회성 이벤트 리스너
   */
  once(eventName, callback, context = null) {
    const onceWrapper = (...args) => {
      callback.call(context, ...args);
      this.off(eventName, onceWrapper);
    };
    
    this.onceEvents.add(onceWrapper);
    return this.on(eventName, onceWrapper, context);
  }

  /**
   * 이벤트 리스너 제거
   */
  off(eventName, callback = null) {
    if (!this.events.has(eventName)) return;

    const listeners = this.events.get(eventName);

    if (callback) {
      // 특정 콜백 제거
      const index = listeners.findIndex(listener => listener.callback === callback);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    } else {
      // 모든 리스너 제거
      this.events.set(eventName, []);
    }

    // 빈 이벤트 정리
    if (listeners.length === 0) {
      this.events.delete(eventName);
    }
  }

  /**
   * 이벤트 발생
   */
  emit(eventName, ...args) {
    if (!this.events.has(eventName)) return 0;

    const listeners = [...this.events.get(eventName)]; // 복사본으로 안전성 확보
    let executedCount = 0;

    listeners.forEach(({ callback, context }) => {
      try {
        callback.call(context, ...args);
        executedCount++;
      } catch (error) {
        console.error(`EventBus error in "${eventName}":`, error);
      }
    });

    return executedCount;
  }

  /**
   * 비동기 이벤트 발생
   */
  async emitAsync(eventName, ...args) {
    if (!this.events.has(eventName)) return 0;

    const listeners = [...this.events.get(eventName)];
    let executedCount = 0;

    for (const { callback, context } of listeners) {
      try {
        await callback.call(context, ...args);
        executedCount++;
      } catch (error) {
        console.error(`EventBus async error in "${eventName}":`, error);
      }
    }

    return executedCount;
  }

  /**
   * 이벤트 존재 확인
   */
  hasEvent(eventName) {
    return this.events.has(eventName) && this.events.get(eventName).length > 0;
  }

  /**
   * 리스너 수 조회
   */
  listenerCount(eventName) {
    return this.events.has(eventName) ? this.events.get(eventName).length : 0;
  }

  /**
   * 모든 이벤트 정리
   */
  clear() {
    this.events.clear();
    this.onceEvents.clear();
  }

  /**
   * 디버그 정보
   */
  getDebugInfo() {
    const info = {
      totalEvents: this.events.size,
      events: {}
    };

    for (const [eventName, listeners] of this.events) {
      info.events[eventName] = {
        listenerCount: listeners.length,
        listeners: listeners.map((l, i) => ({
          index: i,
          hasContext: !!l.context,
          functionName: l.callback.name || 'anonymous'
        }))
      };
    }

    return info;
  }
}

// 글로벌 이벤트 버스 인스턴스
export const eventBus = new EventBus();

// 표준 이벤트 상수
export const EVENTS = {
  // 데이터 관련
  DATA_LOADING_START: 'data:loading:start',
  DATA_LOADING_END: 'data:loading:end', 
  DATA_ERROR: 'data:error',
  DATA_UPDATED: 'data:updated',
  
  // UI 관련
  PAGE_CHANGE: 'ui:page:change',
  COMPONENT_READY: 'ui:component:ready',
  COMPONENT_ERROR: 'ui:component:error',
  
  // 시스템 관련
  APP_READY: 'system:app:ready',
  REFRESH_TRIGGERED: 'system:refresh:triggered',
  CACHE_CLEARED: 'system:cache:cleared',
  
  // 차트 관련
  CHART_CREATED: 'chart:created',
  CHART_UPDATED: 'chart:updated',
  CHART_ERROR: 'chart:error'
};