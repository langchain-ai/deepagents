/**
 * Applies an async worker over items with bounded concurrency.
 *
 * The result array preserves input order.
 */
export async function mapWithConcurrency<T, U>(
  items: T[],
  concurrency: number,
  worker: (item: T, index: number) => Promise<U>
): Promise<U[]> {
  if (items.length === 0) {
    return [];
  }
  const limit = Math.max(1, Math.floor(concurrency));
  const results = new Array<U>(items.length);
  let next = 0;

  async function runWorker(): Promise<void> {
    while (true) {
      const index = next;
      next += 1;
      if (index >= items.length) {
        return;
      }
      results[index] = await worker(items[index], index);
    }
  }

  const workerCount = Math.min(limit, items.length);
  await Promise.all(Array.from({ length: workerCount }, () => runWorker()));
  return results;
}
