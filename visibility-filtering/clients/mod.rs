use thrift::protocol::{TFieldIdentifier, TInputProtocol, TType};

pub mod about_this_account_client;
pub mod article_client;
pub mod gizmoduck_client;
pub mod socialgraph_client;
pub mod trusted_friends_client;
pub mod user_location_client;
pub mod wingman_client;

fn read_fields(
    i_prot: &mut dyn TInputProtocol,
    mut read_field: impl FnMut(&mut dyn TInputProtocol, &TFieldIdentifier) -> thrift::Result<bool>,
) -> thrift::Result<()> {
    i_prot.read_struct_begin()?;
    loop {
        let field = i_prot.read_field_begin()?;
        if field.field_type == TType::Stop {
            break;
        }
        if !read_field(i_prot, &field)? {
            i_prot.skip(field.field_type)?;
        }
        i_prot.read_field_end()?;
    }
    i_prot.read_struct_end()
}
