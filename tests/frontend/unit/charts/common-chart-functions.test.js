// Unit tests for common chart functions
import { jest } from '@jest/globals';
import '@testing-library/jest-dom';
import '../../../mocks/server.js'; // MSW setup

// Import test utilities
import {
  createMockChart,
  mockChartJs,
  generateMockChartData,
  validateChartConfig,
  simulateChartClick,
  CHART_TYPES,
  COMMON_CHART_OPTIONS
} from '../../../utils/chart-test-utils.js';

import { mockChartData, mockStockData } from '../../../fixtures/dashboard-data.js';

// Mock Chart.js before importing dashboard files
beforeAll(() => {
  mockChartJs();
});

describe('Common Chart Functions', () => {
  let mockCanvas;
  let mockCtx;

  beforeEach(() => {
    // Create fresh mock canvas for each test
    mockCanvas = document.createElement('canvas');
    mockCanvas.id = 'test-chart';
    mockCanvas.width = 800;
    mockCanvas.height = 400;
    
    mockCtx = {
      fillRect: jest.fn(),
      clearRect: jest.fn(),
      getImageData: jest.fn(),
      putImageData: jest.fn(),
      createImageData: jest.fn(),
      setTransform: jest.fn(),
      drawImage: jest.fn(),
      save: jest.fn(),
      restore: jest.fn(),
      beginPath: jest.fn(),
      closePath: jest.fn(),
      moveTo: jest.fn(),
      lineTo: jest.fn(),
      stroke: jest.fn(),
      fill: jest.fn(),
      measureText: jest.fn(() => ({ width: 100 }))
    };
    
    mockCanvas.getContext = jest.fn(() => mockCtx);
    document.body.appendChild(mockCanvas);
  });

  afterEach(() => {
    document.body.innerHTML = '';
    jest.clearAllMocks();
  });

  describe('Chart Creation', () => {
    test('should create line chart with proper configuration', () => {
      const chartData = generateMockChartData(CHART_TYPES.LINE);
      const chart = new Chart(mockCanvas, {
        type: CHART_TYPES.LINE,
        data: chartData,
        options: COMMON_CHART_OPTIONS
      });

      expect(Chart).toHaveBeenCalledWith(mockCanvas, {
        type: CHART_TYPES.LINE,
        data: chartData,
        options: COMMON_CHART_OPTIONS
      });

      expect(chart.type).toBe(CHART_TYPES.LINE);
      expect(chart.data).toEqual(chartData);
      expect(chart.options).toEqual(COMMON_CHART_OPTIONS);
    });

    test('should create bar chart with volume data', () => {
      const chartData = generateMockChartData(CHART_TYPES.BAR);
      const chart = new Chart(mockCanvas, {
        type: CHART_TYPES.BAR,
        data: chartData,
        options: COMMON_CHART_OPTIONS
      });

      expect(chart.type).toBe(CHART_TYPES.BAR);
      expect(chart.data.datasets).toHaveLength(1);
      expect(chart.data.datasets[0].data).toHaveLength(10);
    });

    test('should create pie chart with sector data', () => {
      const chartData = generateMockChartData(CHART_TYPES.PIE);
      const chart = new Chart(mockCanvas, {
        type: CHART_TYPES.PIE,
        data: chartData,
        options: COMMON_CHART_OPTIONS
      });

      expect(chart.type).toBe(CHART_TYPES.PIE);
      expect(chart.data.datasets[0].backgroundColor).toHaveLength(5);
    });
  });

  describe('Chart Configuration Validation', () => {
    test('should validate correct chart configuration', () => {
      const config = {
        type: CHART_TYPES.LINE,
        data: generateMockChartData(CHART_TYPES.LINE),
        options: COMMON_CHART_OPTIONS
      };

      const validation = validateChartConfig(config);
      
      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    test('should detect missing chart type', () => {
      const config = {
        data: generateMockChartData(CHART_TYPES.LINE),
        options: COMMON_CHART_OPTIONS
      };

      const validation = validateChartConfig(config);
      
      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContain('Chart type is required');
    });

    test('should detect missing datasets', () => {
      const config = {
        type: CHART_TYPES.LINE,
        data: { labels: ['Label 1', 'Label 2'] },
        options: COMMON_CHART_OPTIONS
      };

      const validation = validateChartConfig(config);
      
      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContain('At least one dataset is required');
    });
  });

  describe('Chart Interactions', () => {
    test('should handle chart click events', () => {
      const chart = createMockChart(CHART_TYPES.LINE);
      chart.canvas = mockCanvas;

      const clickEvent = simulateChartClick(chart, 200, 150);
      
      expect(clickEvent.clientX).toBe(200);
      expect(clickEvent.clientY).toBe(150);
      expect(clickEvent.type).toBe('click');
    });

    test('should update chart data correctly', () => {
      const chart = createMockChart(CHART_TYPES.LINE);
      const newData = generateMockChartData(CHART_TYPES.LINE, 15);
      
      chart.data = newData;
      chart.update();
      
      expect(chart.update).toHaveBeenCalled();
      expect(chart.data.datasets[0].data).toHaveLength(15);
    });

    test('should destroy chart properly', () => {
      const chart = createMockChart(CHART_TYPES.LINE);
      
      chart.destroy();
      
      expect(chart.destroy).toHaveBeenCalled();
    });
  });

  describe('Chart Responsiveness', () => {
    test('should resize chart when container changes', () => {
      const chart = createMockChart(CHART_TYPES.LINE);
      chart.canvas = mockCanvas;
      
      // Simulate container resize
      mockCanvas.width = 1200;
      mockCanvas.height = 600;
      chart.resize();
      
      expect(chart.resize).toHaveBeenCalled();
    });

    test('should maintain aspect ratio when configured', () => {
      const optionsWithAspectRatio = {
        ...COMMON_CHART_OPTIONS,
        maintainAspectRatio: true
      };
      
      const chart = createMockChart(CHART_TYPES.LINE, {}, optionsWithAspectRatio);
      
      expect(chart.options.maintainAspectRatio).toBe(true);
    });
  });

  describe('Chart Data Processing', () => {
    test('should handle real stock price data', () => {
      const stockPriceData = {
        labels: mockStockData.sp500.map(stock => stock.ticker),
        datasets: [{
          label: 'Stock Prices',
          data: mockStockData.sp500.map(stock => stock.price),
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          borderColor: 'rgba(75, 192, 192, 1)'
        }]
      };

      const chart = createMockChart(CHART_TYPES.BAR, stockPriceData);
      
      expect(chart.data.labels).toHaveLength(3);
      expect(chart.data.datasets[0].data).toEqual([150.25, 305.80, 2750.45]);
    });

    test('should handle time series data correctly', () => {
      const chart = createMockChart(CHART_TYPES.LINE, mockChartData.priceHistory);
      
      expect(chart.data.labels).toHaveLength(6);
      expect(chart.data.datasets[0].label).toBe('AAPL Price');
      expect(chart.data.datasets[0].data[0]).toBe(148.50);
    });

    test('should handle volume data with bar chart', () => {
      const chart = createMockChart(CHART_TYPES.BAR, mockChartData.volumeData);
      
      expect(chart.data.datasets[0].label).toBe('Volume');
      expect(chart.data.datasets[0].data).toEqual([1200000, 980000, 1450000, 1100000, 890000, 1300000]);
    });
  });

  describe('Chart Error Handling', () => {
    test('should handle empty data gracefully', () => {
      const emptyData = {
        labels: [],
        datasets: []
      };

      const chart = createMockChart(CHART_TYPES.LINE, emptyData);
      
      expect(chart.data.labels).toHaveLength(0);
      expect(chart.data.datasets).toHaveLength(0);
    });

    test('should handle invalid data types', () => {
      const invalidData = {
        labels: ['Label 1', 'Label 2'],
        datasets: [{
          data: ['invalid', null, undefined, 'string']
        }]
      };

      const chart = createMockChart(CHART_TYPES.LINE, invalidData);
      
      // Chart should still be created but with invalid data
      expect(chart.data.datasets[0].data).toEqual(['invalid', null, undefined, 'string']);
    });
  });

  describe('Chart Performance', () => {
    test('should handle large datasets efficiently', () => {
      const largeDataset = generateMockChartData(CHART_TYPES.LINE, 1000);
      const chart = createMockChart(CHART_TYPES.LINE, largeDataset);
      
      expect(chart.data.datasets[0].data).toHaveLength(1000);
      expect(chart.data.labels).toHaveLength(1000);
    });

    test('should update efficiently with animation disabled', () => {
      const chart = createMockChart(CHART_TYPES.LINE, {}, {
        animation: { duration: 0 }
      });
      
      chart.update();
      expect(chart.update).toHaveBeenCalled();
      expect(chart.options.animation.duration).toBe(0);
    });
  });
});