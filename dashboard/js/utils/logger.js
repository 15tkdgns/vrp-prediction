/**
 * Dashboard Logging System
 * Provides structured logging for the dashboard application
 */

export class DashboardLogger {
    constructor(name = 'Dashboard') {
        this.name = name;
        this.logLevel = this.getLogLevel();
        this.logs = [];
        this.maxLogs = 1000;
    }

    getLogLevel() {
        const url = new URLSearchParams(window.location.search);
        const debugMode = url.get('debug') === 'true';
        return debugMode ? 'debug' : 'info';
    }

    log(level, message, data = null) {
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            level,
            module: this.name,
            message,
            data
        };

        // Add to internal log storage
        this.logs.push(logEntry);
        if (this.logs.length > this.maxLogs) {
            this.logs.shift();
        }

        // Console output
        const emoji = this.getEmoji(level);
        const consoleMessage = `${emoji} [${this.name}] ${message}`;
        
        switch (level) {
            case 'debug':
                if (this.logLevel === 'debug') {
                    console.debug(consoleMessage, data);
                }
                break;
            case 'info':
                console.info(consoleMessage, data);
                break;
            case 'warn':
                console.warn(consoleMessage, data);
                break;
            case 'error':
                console.error(consoleMessage, data);
                break;
        }

        // Send to monitoring endpoint (if available)
        this.sendToMonitoring(logEntry);
    }

    debug(message, data) {
        this.log('debug', message, data);
    }

    info(message, data) {
        this.log('info', message, data);
    }

    warn(message, data) {
        this.log('warn', message, data);
    }

    error(message, data) {
        this.log('error', message, data);
    }

    getEmoji(level) {
        const emojis = {
            debug: '🔍',
            info: 'ℹ️',
            warn: '⚠️',
            error: '❌'
        };
        return emojis[level] || '📝';
    }

    async sendToMonitoring(logEntry) {
        // Only send errors and warnings to monitoring
        if (!['error', 'warn'].includes(logEntry.level)) {
            return;
        }

        try {
            await fetch('/api/logs', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(logEntry)
            });
        } catch (error) {
            // Silent fail - don't log monitoring failures
        }
    }

    exportLogs() {
        return this.logs;
    }

    clearLogs() {
        this.logs = [];
        this.info('Logs cleared');
    }

    // Performance timing
    startTimer(name) {
        const timerKey = `timer_${name}`;
        performance.mark(`${timerKey}_start`);
        return timerKey;
    }

    endTimer(timerKey) {
        const endMark = `${timerKey}_end`;
        performance.mark(endMark);
        performance.measure(timerKey, `${timerKey}_start`, endMark);
        
        const measures = performance.getEntriesByName(timerKey);
        if (measures.length > 0) {
            const duration = measures[measures.length - 1].duration;
            this.info(`Performance: ${timerKey.replace('timer_', '')} took ${duration.toFixed(2)}ms`);
            return duration;
        }
        return 0;
    }
}

// Create default logger instance
export const logger = new DashboardLogger('Dashboard');

// Export factory function for component-specific loggers
export function createLogger(name) {
    return new DashboardLogger(name);
}