/**
 * Tab Manager
 * Handles tab navigation and content switching
 */

import { createLogger } from '../utils/logger.js';

const tabLogger = createLogger('TabManager');

export class TabManager {
    constructor() {
        this.currentTab = 'overview';
        this.tabs = new Map();
        this.tabHistory = ['overview'];
        this.maxHistory = 10;
        
        this.initializeEventListeners();
        tabLogger.info('Tab Manager initialized');
    }

    initializeEventListeners() {
        // Tab navigation buttons
        document.addEventListener('click', (event) => {
            if (event.target.hasAttribute('data-tab')) {
                event.preventDefault();
                const tabName = event.target.getAttribute('data-tab');
                this.switchToTab(tabName);
            }
        });

        // Keyboard navigation
        document.addEventListener('keydown', (event) => {
            if (event.altKey) {
                switch (event.key) {
                    case '1':
                        event.preventDefault();
                        this.switchToTab('overview');
                        break;
                    case '2':
                        event.preventDefault();
                        this.switchToTab('predictions');
                        break;
                    case '3':
                        event.preventDefault();
                        this.switchToTab('models');
                        break;
                    case '4':
                        event.preventDefault();
                        this.switchToTab('analytics');
                        break;
                    case '5':
                        event.preventDefault();
                        this.switchToTab('monitoring');
                        break;
                }
            }
        });

        // Browser back/forward buttons
        window.addEventListener('popstate', (event) => {
            if (event.state && event.state.tab) {
                this.switchToTab(event.state.tab, false);
            }
        });
    }

    switchToTab(tabName, updateHistory = true) {
        const timer = tabLogger.startTimer(`switch_to_${tabName}`);
        
        try {
            // Validate tab exists
            const tabContent = document.getElementById(`${tabName}-tab`);
            if (!tabContent) {
                throw new Error(`Tab '${tabName}' not found`);
            }

            // Hide current tab
            if (this.currentTab !== tabName) {
                this.hideTab(this.currentTab);
            }

            // Show new tab
            this.showTab(tabName);

            // Update navigation
            this.updateNavigation(tabName);

            // Update history
            if (updateHistory) {
                this.updateHistory(tabName);
            }

            // Update browser URL
            this.updateURL(tabName);

            // Trigger tab events
            this.triggerTabEvents(tabName);

            this.currentTab = tabName;
            tabLogger.info(`Switched to tab: ${tabName}`);
            tabLogger.endTimer(timer);

        } catch (error) {
            tabLogger.error(`Failed to switch to tab: ${tabName}`, error);
            tabLogger.endTimer(timer);
        }
    }

    showTab(tabName) {
        const tabContent = document.getElementById(`${tabName}-tab`);
        if (tabContent) {
            tabContent.classList.remove('d-none');
            tabContent.classList.add('tab-content');
            
            // Add animation
            tabContent.style.opacity = '0';
            setTimeout(() => {
                tabContent.style.transition = 'opacity 0.3s ease-in-out';
                tabContent.style.opacity = '1';
            }, 10);
        }
    }

    hideTab(tabName) {
        const tabContent = document.getElementById(`${tabName}-tab`);
        if (tabContent) {
            tabContent.style.opacity = '0';
            setTimeout(() => {
                tabContent.classList.add('d-none');
                tabContent.classList.remove('tab-content');
            }, 150);
        }
    }

    updateNavigation(activeTab) {
        // Remove active class from all nav items
        const navLinks = document.querySelectorAll('[data-tab]');
        navLinks.forEach(link => {
            link.classList.remove('active');
        });

        // Add active class to current tab
        const activeNavLink = document.querySelector(`[data-tab="${activeTab}"]`);
        if (activeNavLink) {
            activeNavLink.classList.add('active');
        }
    }

    updateHistory(tabName) {
        if (this.tabHistory[this.tabHistory.length - 1] !== tabName) {
            this.tabHistory.push(tabName);
            
            // Limit history size
            if (this.tabHistory.length > this.maxHistory) {
                this.tabHistory.shift();
            }
        }
    }

    updateURL(tabName) {
        const url = new URL(window.location);
        url.searchParams.set('tab', tabName);
        
        window.history.pushState(
            { tab: tabName },
            `SPY Analysis - ${this.getTabTitle(tabName)}`,
            url.toString()
        );
    }

    getTabTitle(tabName) {
        const titles = {
            overview: 'Overview',
            predictions: 'Predictions',
            models: 'Models',
            analytics: 'Analytics',
            monitoring: 'Monitoring'
        };
        return titles[tabName] || 'Dashboard';
    }

    triggerTabEvents(tabName) {
        // Custom tab switch event
        window.dispatchEvent(new CustomEvent('tabSwitch', {
            detail: {
                from: this.currentTab,
                to: tabName,
                timestamp: Date.now()
            }
        }));

        // Tab-specific events
        window.dispatchEvent(new CustomEvent(`tab:${tabName}:show`, {
            detail: { timestamp: Date.now() }
        }));
    }

    // Tab registration for lazy loading
    registerTab(tabName, loadFunction) {
        this.tabs.set(tabName, {
            loaded: false,
            loadFunction
        });
        
        tabLogger.debug(`Tab registered: ${tabName}`);
    }

    async loadTabContent(tabName) {
        const tab = this.tabs.get(tabName);
        if (!tab || tab.loaded) {
            return;
        }

        try {
            tabLogger.info(`Loading content for tab: ${tabName}`);
            await tab.loadFunction();
            tab.loaded = true;
            tabLogger.info(`Tab content loaded: ${tabName}`);
        } catch (error) {
            tabLogger.error(`Failed to load tab content: ${tabName}`, error);
        }
    }

    // Navigation methods
    goBack() {
        if (this.tabHistory.length > 1) {
            this.tabHistory.pop(); // Remove current
            const previousTab = this.tabHistory[this.tabHistory.length - 1];
            this.switchToTab(previousTab, false);
            tabLogger.debug(`Navigated back to: ${previousTab}`);
        }
    }

    getTabFromURL() {
        const url = new URLSearchParams(window.location.search);
        return url.get('tab') || 'overview';
    }

    initializeFromURL() {
        const urlTab = this.getTabFromURL();
        if (urlTab !== this.currentTab) {
            this.switchToTab(urlTab, false);
        }
    }

    // Utility methods
    getCurrentTab() {
        return this.currentTab;
    }

    getTabHistory() {
        return [...this.tabHistory];
    }

    isTabLoaded(tabName) {
        const tab = this.tabs.get(tabName);
        return tab ? tab.loaded : false;
    }

    // Keyboard shortcuts help
    showKeyboardShortcuts() {
        const shortcuts = [
            'Alt+1: Overview',
            'Alt+2: Predictions', 
            'Alt+3: Models',
            'Alt+4: Analytics',
            'Alt+5: Monitoring'
        ];

        // Show shortcuts in console or modal
        tabLogger.info('Keyboard shortcuts:', shortcuts);
        
        // Could also show in a modal or tooltip
        return shortcuts;
    }
}

// Create default tab manager instance
export const tabManager = new TabManager();