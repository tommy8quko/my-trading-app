-- Stores the current (possibly trailed) stop loss per open position.
-- Separate from stop_loss_at_entry on orders (which records the initial stop).
create table if not exists position_stops (
    symbol       text primary key,
    current_stop double precision not null,
    is_initial   boolean not null default true,
    updated_at   timestamptz not null default now()
);
