// MSW server setup for Node.js environment (Jest tests)
import { setupServer } from 'msw/node';
import { handlers } from './handlers.js';

// Setup mock server with handlers
export const server = setupServer(...handlers);

// Establish API mocking before all tests
beforeAll(() => {
  server.listen({
    onUnhandledRequest: 'warn' // Warn about requests not covered by handlers
  });
});

// Reset handlers after each test
afterEach(() => {
  server.resetHandlers();
});

// Clean up after all tests
afterAll(() => {
  server.close();
});

// Helper function to add runtime handlers
export const addMockHandler = (handler) => {
  server.use(handler);
};

// Helper function to simulate network errors
export const simulateNetworkError = (url) => {
  server.use(
    http.get(url, () => {
      return HttpResponse.error();
    })
  );
};

// Helper function to simulate slow responses
export const simulateSlowResponse = (url, delay = 3000) => {
  server.use(
    http.get(url, async () => {
      await new Promise(resolve => setTimeout(resolve, delay));
      return HttpResponse.json({ message: 'Slow response' });
    })
  );
};