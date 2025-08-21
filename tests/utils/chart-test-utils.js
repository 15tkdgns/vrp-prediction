// Chart.js testing utilities for dashboard charts
import { jest } from '@jest/globals';

/**
 * Create a mock Chart.js instance with common methods
 */
export const createMockChart = (type = 'line', data = {}, options = {}) => {
  const mockChart = {
    type,
    data: {
      labels: [],
      datasets: [],
      ...data
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      ...options
    },
    canvas: createMockCanvas(),
    ctx: null,
    
    // Chart.js methods
    update: jest.fn(),
    destroy: jest.fn(),
    render: jest.fn(),
    resize: jest.fn(),
    clear: jest.fn(),
    stop: jest.fn(),
    reset: jest.fn(),
    
    // Data manipulation
    getDatasetMeta: jest.fn(() => ({ data: [] })),
    getDatasetAtEvent: jest.fn(() => []),
    getElementAtEvent: jest.fn(() => []),
    getElementsAtEvent: jest.fn(() => []),
    
    // Export methods
    toBase64Image: jest.fn(() => 'data:image/png;base64,mockimage'),
    generateLegend: jest.fn(() => '<div>Mock Legend</div>'),
    
    // Configuration
    config: {
      type,
      data: data,
      options: options
    }
  };

  // Set up canvas context
  mockChart.ctx = mockChart.canvas.getContext('2d');
  
  return mockChart;
};

/**
 * Mock Chart.js constructor that returns our mock chart
 */
export const mockChartJs = () => {
  global.Chart = jest.fn().mockImplementation((ctx, config) => {
    return createMockChart(config.type, config.data, config.options);
  });

  // Chart.js static methods
  global.Chart.register = jest.fn();
  global.Chart.unregister = jest.fn();
  global.Chart.getChart = jest.fn(() => null);
  global.Chart.defaults = {
    global: {
      defaultFontFamily: 'Arial',
      defaultFontSize: 12
    }
  };

  return global.Chart;
};

/**
 * Generate mock data for different chart types
 */
export const generateMockChartData = (type, pointCount = 10) => {
  const labels = Array.from({ length: pointCount }, (_, i) => `Label ${i + 1}`);
  
  const baseDataset = {
    label: 'Mock Dataset',
    borderWidth: 1,
    backgroundColor: 'rgba(75, 192, 192, 0.2)',
    borderColor: 'rgba(75, 192, 192, 1)'
  };

  switch (type) {
    case 'line':
      return {
        labels,
        datasets: [{
          ...baseDataset,
          data: Array.from({ length: pointCount }, () => Math.random() * 100),
          fill: false
        }]
      };

    case 'bar':
      return {
        labels,
        datasets: [{
          ...baseDataset,
          data: Array.from({ length: pointCount }, () => Math.random() * 100)
        }]
      };

    case 'pie':
    case 'doughnut':
      return {
        labels: labels.slice(0, 5), // Fewer labels for pie charts
        datasets: [{
          data: Array.from({ length: 5 }, () => Math.random() * 100),
          backgroundColor: [
            'rgba(255, 99, 132, 0.2)',
            'rgba(54, 162, 235, 0.2)',
            'rgba(255, 205, 86, 0.2)',
            'rgba(75, 192, 192, 0.2)',
            'rgba(153, 102, 255, 0.2)'
          ]
        }]
      };

    case 'scatter':
      return {
        datasets: [{
          ...baseDataset,
          data: Array.from({ length: pointCount }, () => ({
            x: Math.random() * 100,
            y: Math.random() * 100
          }))
        }]
      };

    case 'candlestick':
      return {
        labels,
        datasets: [{
          label: 'Candlestick Data',
          data: Array.from({ length: pointCount }, () => {
            const open = Math.random() * 100 + 50;
            const close = open + (Math.random() - 0.5) * 10;
            const high = Math.max(open, close) + Math.random() * 5;
            const low = Math.min(open, close) - Math.random() * 5;
            return { open, high, low, close };
          })
        }]
      };

    default:
      return {
        labels,
        datasets: [{ ...baseDataset, data: Array.from({ length: pointCount }, () => Math.random() * 100) }]
      };
  }
};

/**
 * Test helper for chart configuration validation
 */
export const validateChartConfig = (config) => {
  const errors = [];

  if (!config.type) {
    errors.push('Chart type is required');
  }

  if (!config.data) {
    errors.push('Chart data is required');
  } else {
    if (!config.data.datasets || config.data.datasets.length === 0) {
      errors.push('At least one dataset is required');
    }
  }

  if (!config.options) {
    errors.push('Chart options should be defined');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * Simulate chart interactions
 */
export const simulateChartClick = (chart, x = 100, y = 100) => {
  const event = new MouseEvent('click', {
    clientX: x,
    clientY: y,
    bubbles: true
  });

  chart.canvas.dispatchEvent(event);
  return event;
};

export const simulateChartHover = (chart, x = 100, y = 100) => {
  const event = new MouseEvent('mousemove', {
    clientX: x,
    clientY: y,
    bubbles: true
  });

  chart.canvas.dispatchEvent(event);
  return event;
};

/**
 * Test chart responsive behavior
 */
export const testChartResponsiveness = (chart, newWidth = 800, newHeight = 600) => {
  // Mock canvas resize
  chart.canvas.width = newWidth;
  chart.canvas.height = newHeight;
  
  // Trigger resize
  chart.resize();
  
  return {
    resizeCalled: chart.resize.mock.calls.length > 0,
    newDimensions: { width: newWidth, height: newHeight }
  };
};

/**
 * Validate chart accessibility
 */
export const validateChartAccessibility = (chartContainer) => {
  const issues = [];

  // Check for aria-label
  if (!chartContainer.getAttribute('aria-label')) {
    issues.push('Missing aria-label');
  }

  // Check for role
  if (!chartContainer.getAttribute('role')) {
    issues.push('Missing role attribute');
  }

  // Check for canvas alt text
  const canvas = chartContainer.querySelector('canvas');
  if (canvas && !canvas.getAttribute('aria-label')) {
    issues.push('Canvas missing aria-label');
  }

  return {
    isAccessible: issues.length === 0,
    issues
  };
};

/**
 * Performance testing helpers
 */
export const measureChartRenderTime = async (chartFactory) => {
  const startTime = performance.now();
  await chartFactory();
  const endTime = performance.now();
  return endTime - startTime;
};

export const simulateDataUpdate = (chart, newData) => {
  chart.data = { ...chart.data, ...newData };
  chart.update();
  return chart.update.mock.calls.length;
};

/**
 * Helper to wait for chart animations
 */
export const waitForChartAnimation = (chart, timeout = 1000) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(chart);
    }, timeout);
  });
};

/**
 * Chart testing constants
 */
export const CHART_TYPES = {
  LINE: 'line',
  BAR: 'bar',
  PIE: 'pie',
  DOUGHNUT: 'doughnut',
  SCATTER: 'scatter',
  CANDLESTICK: 'candlestick',
  AREA: 'area'
};

export const COMMON_CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      position: 'top'
    },
    tooltip: {
      enabled: true
    }
  },
  scales: {
    y: {
      beginAtZero: true
    }
  }
};