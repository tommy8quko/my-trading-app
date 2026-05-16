-- Stores the current (possibly trailed) stop loss per lot (order).
-- Separate from stop_loss_at_entry on orders (which records the initial stop at entry).
-- Keyed by order_id so positions with multiple lots each get their own editable stop.
create table if not exists position_stops (
    order_id     text primary key,
    current_stop double precision not null,
    is_initial   boolean not null default true,
    updated_at   timestamptz not null default now()
);
