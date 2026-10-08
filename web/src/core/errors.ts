/**
 * Thrown by a method function that has not been implemented (e.g. a stub
 * substituted via METHOD_IMPL_DIR).
 * The application catches it and reports the function as "not implemented"
 * instead of showing a number.
 */
export class NotImplementedError extends Error {
  readonly functionName: string;

  constructor(functionName: string) {
    super(`${functionName} is not implemented yet (see web/docs/MATH_SPEC.md)`);
    this.name = 'NotImplementedError';
    this.functionName = functionName;
  }
}

export function isNotImplemented(error: unknown): error is NotImplementedError {
  return error instanceof NotImplementedError;
}
