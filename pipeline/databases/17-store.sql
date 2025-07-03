-- creates a trigger that decreases the quantity of an item after adding a new order.
DROP TRIGGER IF EXISTS order_update;
CREATE TRIGGER order_update
AFTER INSERT
ON orders FOR EACH ROW
    UPDATE items
    SET items.quantity = items.quantity - NEW.number
    WHERE items.name = NEW.item_name;
