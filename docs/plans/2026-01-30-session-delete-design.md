# Session Delete Feature Design

## Overview

Add the ability to delete chat sessions from the sidebar with a hover-to-reveal trash icon and confirmation dialog.

## User Flow

1. User hovers over a session in the sidebar
2. Trash icon appears on the right side of the session
3. User clicks the trash icon
4. Confirmation dialog appears: "Delete this chat? This cannot be undone."
5. User clicks "Delete" to confirm (or "Cancel" to abort)
6. Session is deleted via API
7. Session disappears from sidebar
8. If the deleted session was currently active, reset to "New Chat" state

## Technical Design

### Data Flow

```
User hovers session → Trash icon appears
User clicks trash → Confirmation dialog opens
User confirms → DELETE /api/v1/chat/sessions/{id}
Backend deletes → 204 No Content
Frontend removes session from list
If deleted session was active → Reset to "New Chat"
```

### Backend

No changes needed. The following endpoint already exists and is fully implemented:

- `DELETE /api/v1/chat/sessions/{session_id}` - Returns 204 on success, 404 if not found

### Frontend Changes

#### 1. Add shadcn AlertDialog component

```bash
npx shadcn@latest add alert-dialog
```

#### 2. Modify `frontend/src/hooks/useSessions.ts`

Add `deleteSession` function:

```tsx
const deleteSession = useCallback(async (sessionId: string) => {
  const token = await getToken();
  if (!token) throw new Error("No auth token");

  const res = await fetch(`${BACKEND_URL}/chat/sessions/${sessionId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!res.ok) throw new Error("Failed to delete session");

  // Optimistically remove from local cache
  mutate(
    (current) => current ? {
      ...current,
      sessions: current.sessions.filter(s => s.id !== sessionId),
      total: current.total - 1,
    } : current,
    { revalidate: false }
  );
}, [getToken, mutate]);
```

Return `deleteSession` from the hook.

#### 3. Modify `frontend/src/components/chat/Sidebar.tsx`

Add new prop:
```tsx
onDeleteSession?: (sessionId: string) => void;
```

Wrap each session button in a group container with hover-revealed trash icon:

```tsx
<div key={session.id} className="group relative">
  <Button
    variant="ghost"
    className={cn(
      "w-full justify-start gap-2 font-normal truncate transition-all pr-8",
      currentSessionId === session.id
        ? "bg-white/10 text-white"
        : "text-white/60 hover:text-white hover:bg-white/5"
    )}
    onClick={() => onSelectSession?.(session.id)}
  >
    <MessageSquare className="h-4 w-4 flex-shrink-0 opacity-70" />
    <span className="truncate">{session.name}</span>
  </Button>

  <button
    className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-white/10 transition-opacity"
    onClick={(e) => {
      e.stopPropagation();
      onDeleteSession?.(session.id);
    }}
  >
    <Trash2 className="h-4 w-4 text-white/40 hover:text-red-400 transition-colors" />
  </button>
</div>
```

#### 4. Modify `frontend/src/components/chat/ChatLayout.tsx`

Add state for the delete confirmation dialog:
```tsx
const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
```

Add delete handler:
```tsx
const handleDeleteSession = useCallback(async () => {
  if (!sessionToDelete) return;

  try {
    await deleteSession(sessionToDelete);

    // If we deleted the active session, reset to new chat
    if (sessionToDelete === currentSessionId) {
      setCurrentSessionId(null);
      dispatchNewChatEvent();
    }
  } catch (err) {
    console.error("Failed to delete session:", err);
  } finally {
    setSessionToDelete(null);
  }
}, [sessionToDelete, currentSessionId, deleteSession]);
```

Pass to Sidebar:
```tsx
<Sidebar
  ...
  onDeleteSession={(id) => setSessionToDelete(id)}
/>
```

Add AlertDialog:
```tsx
<AlertDialog open={!!sessionToDelete} onOpenChange={(open) => !open && setSessionToDelete(null)}>
  <AlertDialogContent>
    <AlertDialogHeader>
      <AlertDialogTitle>Delete this chat?</AlertDialogTitle>
      <AlertDialogDescription>
        This will permanently delete this conversation and all its messages. This action cannot be undone.
      </AlertDialogDescription>
    </AlertDialogHeader>
    <AlertDialogFooter>
      <AlertDialogCancel>Cancel</AlertDialogCancel>
      <AlertDialogAction onClick={handleDeleteSession} className="bg-red-600 hover:bg-red-700">
        Delete
      </AlertDialogAction>
    </AlertDialogFooter>
  </AlertDialogContent>
</AlertDialog>
```

## Files Changed

| File | Change |
|------|--------|
| `frontend/src/components/ui/alert-dialog.tsx` | New file (shadcn component) |
| `frontend/src/hooks/useSessions.ts` | Add `deleteSession` function |
| `frontend/src/components/chat/Sidebar.tsx` | Add hover trash icon, `onDeleteSession` prop |
| `frontend/src/components/chat/ChatLayout.tsx` | Add delete confirmation dialog and handler |

## Testing

1. Hover over a session - trash icon should appear
2. Click trash icon - confirmation dialog should open
3. Click Cancel - dialog closes, nothing deleted
4. Click Delete - session removed from list
5. Delete the currently active session - should reset to "New Chat"
6. Delete a non-active session - current view unchanged
