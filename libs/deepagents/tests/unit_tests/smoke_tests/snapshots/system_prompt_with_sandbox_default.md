## Shell paths vs. virtual paths

The `execute` tool runs commands in the host shell and can only access files that exist on the host filesystem.

Some paths returned by the file tools are virtual mounts:

- If a virtual mount has a host path mapping, replace its virtual prefix with the host prefix when running shell commands.
- If a virtual mount does not have a host path mapping, it is not accessible from the shell. Use the file tools listed above to interact with those files.

Do not assume that a path returned by a file tool can be used directly in a shell command.

Virtual mounts without a host path mapping (not accessible from the shell):
- `/common/`
